# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        # NPU adaptation: eager attention for compatibility; float16 since bf16 is not supported
        extra_kwargs["attn_implementation"] = "eager"
        extra_kwargs["torch_dtype"] = torch.float16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        self._visual_cache_initialized = False

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _ensure_visual_cache(self):
        """Lazily pre-compute and cache visual encoder static values.

        Called once on first inference when model is already on the target device.
        Bypasses torch.linspace / .tolist() / .item() that block torchair compilation.
        Also monkey-patches visual attention to use reshape instead of dynamic split.
        """
        if self._visual_cache_initialized:
            return

        visual = self.model.model.visual
        # Fixed grid_thw for the dataset: 4 images, each 16x16
        grid_thw = torch.tensor(
            [[1, 16, 16]] * 4, dtype=torch.long, device=visual.patch_embed.proj.weight.device
        )

        # 1. Position embeddings (from fast_pos_embed_interpolate)
        self._cached_visual_pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)

        # 2. Rotary position embeddings (from rot_pos_emb)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        seq_len = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        self._cached_visual_pe_cos = emb.cos()
        self._cached_visual_pe_sin = emb.sin()

        # 3. cu_seqlens for attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self._cached_visual_cu_seqlens = cu_seqlens

        # 4. split_sizes for get_image_features output
        self._cached_visual_split_sizes = (
            grid_thw.prod(-1) // visual.spatial_merge_size**2
        ).tolist()

        # 5. Monkey-patch visual attention to use reshape instead of dynamic split
        self._patch_visual_attention(visual)

        self._visual_cache_initialized = True
        logger.info("Visual encoder static values cached")

    def _patch_visual_attention(self, visual):
        """Replace Qwen3VLVisionAttention.forward with a reshape-based version.

        The original uses torch.split(lengths.tolist(), dim=2) which creates
        data-dependent symbolic shapes. We replace it with reshape to static
        [num_images, num_heads, tokens_per_image, head_dim].

        Uses explicit matmul + softmax(float32) + matmul to match the original
        eager attention path exactly (no SDPA which has different numerics).
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

        num_images = 4
        tokens_per_image = 256  # 16*16=256 per image, spatial merge happens in merger layer

        for blk in visual.blocks:
            attn = blk.attn
            num_heads = attn.num_heads
            head_dim = attn.head_dim
            scaling = attn.scaling
            proj = attn.proj
            qkv = attn.qkv

            def _make_forward(nh, hd, sc, pr, qkvl, n_img, tpi):
                def _forward(self_attn, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
                    seq_length = hidden_states.shape[0]
                    q, k, v = (
                        qkvl(hidden_states)
                        .reshape(seq_length, 3, nh, hd)
                        .permute(1, 0, 2, 3)
                        .unbind(0)
                    )
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

                    # Reshape to batched format: [seq, nh, hd] → [n_img, nh, tpi, hd]
                    q = q.reshape(n_img, tpi, nh, hd).permute(0, 2, 1, 3)
                    k = k.reshape(n_img, tpi, nh, hd).permute(0, 2, 1, 3)
                    v = v.reshape(n_img, tpi, nh, hd).permute(0, 2, 1, 3)

                    # Explicit eager attention: matmul + softmax(float32) + matmul
                    # Matches original eager_attention_forward exactly
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * sc
                    attn_weights = torch.nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(q.dtype)
                    attn_output = torch.matmul(attn_weights, v)

                    # Reshape back: [n_img, nh, tpi, hd] → [seq, hidden]
                    attn_output = attn_output.permute(0, 2, 1, 3).reshape(seq_length, -1).contiguous()
                    attn_output = pr(attn_output)
                    return attn_output

                return _forward

            attn.forward = _make_forward(
                num_heads, head_dim, scaling, proj, qkv,
                num_images, tokens_per_image
            ).__get__(attn, type(attn))

        logger.info("Patched visual attention with reshape-based forward")

    def _compiled_visual_forward(self, pixel_values: torch.Tensor):
        """Visual encoder forward using cached position embeddings (compilable with torchair)."""
        visual = self.model.model.visual

        hidden_states = visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self._cached_visual_pos_embeds.to(
            hidden_states.device, hidden_states.dtype
        )
        position_embeddings = (
            self._cached_visual_pe_cos.to(hidden_states.device, hidden_states.dtype),
            self._cached_visual_pe_sin.to(hidden_states.device, hidden_states.dtype),
        )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self._cached_visual_cu_seqlens.to(hidden_states.device),
                position_embeddings=position_embeddings,
            )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = visual.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def _compiled_visual_forward_p1(self, pixel_values: torch.Tensor):
        """Visual encoder part 1: patch_embed + pos embed only (no blocks)."""
        visual = self.model.model.visual
        hidden_states = visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self._cached_visual_pos_embeds.to(
            hidden_states.device, hidden_states.dtype
        )
        return hidden_states

    def _compiled_visual_forward_block0(self, hidden_states: torch.Tensor):
        """Single block 0 only."""
        visual = self.model.model.visual
        position_embeddings = (
            self._cached_visual_pe_cos.to(hidden_states.device, hidden_states.dtype),
            self._cached_visual_pe_sin.to(hidden_states.device, hidden_states.dtype),
        )
        cu_seqlens = self._cached_visual_cu_seqlens.to(hidden_states.device)
        hidden_states = visual.blocks[0](
            hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
        )
        return hidden_states

    def _compiled_visual_forward_p2(self, hidden_states: torch.Tensor, deepstack_feature_lists: list):
        """Visual encoder part 2: blocks 1-23 + merger."""
        visual = self.model.model.visual
        position_embeddings = (
            self._cached_visual_pe_cos.to(hidden_states.device, hidden_states.dtype),
            self._cached_visual_pe_sin.to(hidden_states.device, hidden_states.dtype),
        )
        cu_seqlens = self._cached_visual_cu_seqlens.to(hidden_states.device)
        for layer_num in range(1, len(visual.blocks)):
            hidden_states = visual.blocks[layer_num](
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
            )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(visual.deepstack_merger_list[idx](hidden_states))
        hidden_states = visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def _preprocess_vl_input(self, vl_input: dict) -> dict:
        """Run all data-dependent preprocessing (image encoding, embedding, position_ids,
        causal mask, RoPE embeddings, and visual indices).

        All operations here are data-dependent and must run eagerly (not compiled).
        """
        from transformers.masking_utils import create_causal_mask

        qwen3vl_model = self.model.model  # Qwen3VLModel
        lm = self.model.model.language_model

        # 1. Text embedding
        inputs_embeds = qwen3vl_model.get_input_embeddings()(vl_input["input_ids"])

        # 2. Image encoding (use compiled visual forward with cached statics)
        self._ensure_visual_cache()
        pixel_values = vl_input["pixel_values"].to(self.model.model.visual.dtype)
        hidden_states = self._compiled_visual_forward_p1(pixel_values)
        hidden_states = self._compiled_visual_forward_block0(hidden_states)
        raw_embeds, deepstack_image_embeds = self._compiled_visual_forward_p2(hidden_states, [])
        image_embeds_list = torch.split(raw_embeds, self._cached_visual_split_sizes)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        # 3. Scatter image embeddings into text embedding
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds)

        # 4. Visual position masks and deepstack features
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

        # 5. Compute position_ids (data-dependent)
        position_ids, rope_deltas = qwen3vl_model.get_rope_index(
            vl_input["input_ids"],
            image_grid_thw=vl_input["image_grid_thw"],
            attention_mask=vl_input["attention_mask"],
        )

        # 6. Pre-compute items needed by the decoder loop
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        causal_mask = create_causal_mask(
            config=lm.config,
            input_embeds=inputs_embeds,
            attention_mask=vl_input["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        position_embeddings = lm.rotary_emb(inputs_embeds, position_ids)

        visual_indices = visual_pos_masks[0].nonzero().squeeze(-1)

        return {
            "inputs_embeds": inputs_embeds,
            "causal_mask": causal_mask,
            "text_position_ids": text_position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "visual_indices": visual_indices,
        }

    def _language_model_forward(self, **kwargs) -> torch.Tensor:
        """Run the decoder loop only (compilable with torchair).

        All data-dependent operations are pre-computed in _preprocess_vl_input.
        Returns pre-norm hidden states (no final norm).
        """
        inputs_embeds = kwargs["inputs_embeds"]
        causal_mask = kwargs["causal_mask"]
        text_position_ids = kwargs["text_position_ids"]
        cache_position = kwargs["cache_position"]
        position_embeddings = kwargs["position_embeddings"]
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
        visual_indices = kwargs.get("visual_indices")

        lm = self.model.model.language_model
        hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(lm.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                visual_embed = deepstack_visual_embeds[layer_idx].to(
                    hidden_states.device, hidden_states.dtype
                )
                idx = visual_indices.unsqueeze(0).unsqueeze(-1).expand(
                    -1, -1, hidden_states.shape[-1]
                )
                current = torch.gather(hidden_states, 1, idx)
                updated = current + visual_embed.unsqueeze(0)
                hidden_states = hidden_states.scatter(1, idx, updated)

        # Return pre-norm hidden states (no final norm)
        return hidden_states

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}

        # Step 1: data-dependent preprocessing (eager, not compiled)
        lm_kwargs = self._preprocess_vl_input(vl_input)

        # Step 2: language model (compilable with torchair)
        hidden_states = self._language_model_forward(**lm_kwargs)

        # Step 3: output processing
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
