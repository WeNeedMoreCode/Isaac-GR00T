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
import os
import time

import torch
import torch.nn as nn
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
            local_files_only=True,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self._ffn_split4_done = False
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

    def _apply_ffn_split4(self):
        """Split large FFN MatMul (2048→6144) into 4 smaller ones (2048→1536).

        Improves NPU cube utilization. Mathematically equivalent to the original:
          original: down_proj(silu(gate_proj(x)) * up_proj(x))
          split4:   down_proj(cat([silu(g_i(x)) * u_i(x) for i in range(4)]))
        """
        import torch.nn as nn

        num_splits = 4
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            mlp = layer.mlp
            gate_w = mlp.gate_proj.weight  # [6144, 2048]
            up_w = mlp.up_proj.weight      # [6144, 2048]
            gate_b = mlp.gate_proj.bias
            up_b = mlp.up_proj.bias

            in_features = gate_w.shape[1]
            out_features = gate_w.shape[0]
            chunk_size = out_features // num_splits

            if out_features % num_splits != 0:
                logger.warning(
                    f"FFN intermediate_size={out_features} not divisible by {num_splits}, skip split4"
                )
                return

            # Split weights into chunks
            gate_chunks = gate_w.chunk(num_splits, dim=0)
            up_chunks = up_w.chunk(num_splits, dim=0)
            gate_bias_chunks = gate_b.chunk(num_splits, dim=0) if gate_b is not None else [None] * num_splits
            up_bias_chunks = up_b.chunk(num_splits, dim=0) if up_b is not None else [None] * num_splits

            # Create split Linear layers
            gate_linears = nn.ModuleList()
            up_linears = nn.ModuleList()
            for i in range(num_splits):
                g = nn.Linear(in_features, chunk_size, bias=gate_bias_chunks[i] is not None)
                g.weight = nn.Parameter(gate_chunks[i].clone())
                if gate_bias_chunks[i] is not None:
                    g.bias = nn.Parameter(gate_bias_chunks[i].clone())
                gate_linears.append(g)

                u = nn.Linear(in_features, chunk_size, bias=up_bias_chunks[i] is not None)
                u.weight = nn.Parameter(up_chunks[i].clone())
                if up_bias_chunks[i] is not None:
                    u.bias = nn.Parameter(up_bias_chunks[i].clone())
                up_linears.append(u)

            # Move to same device/dtype as original
            gate_linears = gate_linears.to(gate_w.device, gate_w.dtype)
            up_linears = up_linears.to(up_w.device, up_w.dtype)

            # Replace mlp.forward with dual-path version (original + split4, compare)
            act_fn = mlp.act_fn
            orig_gate = mlp.gate_proj
            orig_up = mlp.up_proj
            orig_down = mlp.down_proj

            def _make_split_forward(g_lins, u_lins, act, down):
                def _forward(self_mlp, x):
                    chunks = []
                    for g, u in zip(g_lins, u_lins):
                        chunks.append(act(g(x)) * u(x))
                    return down(torch.cat(chunks, dim=-1))
                return _forward

            mlp.gate_linears = gate_linears
            mlp.up_linears = up_linears
            mlp.forward = _make_split_forward(
                gate_linears, up_linears, act_fn, orig_down
            ).__get__(mlp, type(mlp))

            # Free original large weights
            del mlp.gate_proj
            del mlp.up_proj

        logger.info(
            f"Applied FFN split4 (dual-path verify): {out_features}→{chunk_size} x{num_splits} "
            f"across {len(self.model.language_model.layers)} layers"
        )

    def _ensure_visual_cache(self):
        """Lazily pre-compute and cache visual encoder static values.

        Called once on first inference when model is already on the target device.
        Also applies FFN split4 after model is on NPU (avoids weight format divergence).
        """
        if self._visual_cache_initialized:
            return

        # Apply FFN split4 AFTER model is on NPU (weights already format-converted)
        if not self._ffn_split4_done and not getattr(self, '_skip_ffn_split4', False):
            self._apply_ffn_split4()
            self._ffn_split4_done = True

        visual = self.model.model.visual
        # Fixed grid_thw for the dataset: 4 images, each 16x16
        grid_thw = torch.tensor(
            [[1, 16, 16]] * 4, dtype=torch.long, device=visual.patch_embed.proj.weight.device
        )

        # RC device: patch rot_pos_emb to avoid aicpu ops (.max().item(), .prod().sum().item())
        if getattr(self, '_is_rc', None) is None:
            try:
                from npu_utils import _is_rc_device
                self._is_rc = _is_rc_device()
            except ImportError:
                self._is_rc = False

        if self._is_rc:
            _orig_rot_pos_emb = visual.rot_pos_emb

            def _rot_pos_emb_cpu_safe(grid_thw_tensor):
                orig_device = grid_thw_tensor.device
                result = _orig_rot_pos_emb(grid_thw_tensor.cpu())
                return result.to(orig_device)

            visual.rot_pos_emb = _rot_pos_emb_cpu_safe

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
                    was_3d = hidden_states.ndim == 3
                    seq_length = hidden_states.shape[-2]
                    # QKV projection: sees original ndim so torchair uses 3D kernels when applicable
                    qkv_out = qkvl(hidden_states)
                    # Flatten for attention computation (reshape/rotary operate on 2D)
                    q, k, v = (
                        qkv_out.reshape(seq_length, 3, nh, hd)
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
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * sc
                    attn_weights = torch.nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(q.dtype)
                    attn_output = torch.matmul(attn_weights, v)

                    # Reshape back: [n_img, nh, tpi, hd] → [seq, hidden]
                    attn_output = attn_output.permute(0, 2, 1, 3).reshape(seq_length, -1).contiguous()
                    # Output projection: restore 3D so torchair uses 3D kernels
                    if was_3d:
                        attn_output = attn_output.unsqueeze(0)
                    attn_output = pr(attn_output)
                    return attn_output

                return _forward

            attn.forward = _make_forward(
                num_heads, head_dim, scaling, proj, qkv,
                num_images, tokens_per_image
            ).__get__(attn, type(attn))

        logger.info("Patched visual attention with reshape-based forward")

    def _conv3d_as_linear(self, pixel_values: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Replace Conv3d(kernel=stride, no padding) with reshape + Linear for NPU compatibility."""
        in_c = proj.in_channels
        out_c = proj.out_channels
        kt, kp, _ = proj.kernel_size
        N = pixel_values.shape[0]
        x_flat = pixel_values.reshape(N, in_c * kt * kp * kp).to(dtype=proj.weight.dtype)
        # Use matmul + bias instead of nn.Linear to avoid creating extra parameters
        out = x_flat @ proj.weight.data.reshape(out_c, -1).T
        if proj.bias is not None:
            out = out + proj.bias.data
        return out.view(-1, out_c)

    def _compiled_visual_forward(self, pixel_values: torch.Tensor):
        """Visual encoder forward using cached position embeddings (compilable with torchair).

        Uses 3D tensors (fake batch dim) for better precision when compiled with torchair.
        """
        visual = self.model.model.visual

        # Conv3D strategy (priority: env var > script flag > RC auto-detect):
        if getattr(self, '_use_conv3d_replacement', None) is None:
            if os.environ.get("_GR00T_NATIVE_CONV3D") == "1":
                self._use_conv3d_replacement = False
            elif getattr(self, '_force_conv3d_replace', False):
                self._use_conv3d_replacement = True
            else:
                try:
                    from npu_utils import _is_rc_device
                    self._use_conv3d_replacement = _is_rc_device()
                except ImportError:
                    self._use_conv3d_replacement = False

        if self._use_conv3d_replacement:
            hidden_states = self._conv3d_as_linear(pixel_values, visual.patch_embed.proj)
        else:
            hidden_states = visual.patch_embed(pixel_values)

        hidden_states = hidden_states + self._cached_visual_pos_embeds.to(
            hidden_states.device, hidden_states.dtype
        )
        position_embeddings = (
            self._cached_visual_pe_cos.to(hidden_states.device, hidden_states.dtype),
            self._cached_visual_pe_sin.to(hidden_states.device, hidden_states.dtype),
        )
        cu_seqlens = self._cached_visual_cu_seqlens.to(hidden_states.device)

        # Use 3D tensors for better torchair compiled precision
        hidden_states = hidden_states.unsqueeze(0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = visual.deepstack_merger_list[idx](hidden_states.squeeze(0))
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = visual.merger(hidden_states)
        return hidden_states.squeeze(0), deepstack_feature_lists

    def _preprocess_vl_input(self, vl_input: dict) -> dict:
        """Preprocess VL input: text embedding, image encoding, position/mask/RoPE computation.

        Compilable with torchair. Non-compilable operations (visual cache init, nonzero)
        are pre-computed in forward() and passed via vl_input.
        """
        from transformers.masking_utils import create_causal_mask

        qwen3vl_model = self.model.model
        lm = self.model.model.language_model

        # 1. Text embedding
        inputs_embeds = qwen3vl_model.get_input_embeddings()(vl_input["input_ids"])

        # 2. Image encoding
        pixel_values = vl_input["pixel_values"].to(qwen3vl_model.visual.dtype)
        raw_embeds, deepstack_image_embeds = self._compiled_visual_forward(pixel_values)
        image_embeds = raw_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # 3. Causal mask + cache position
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        causal_mask = create_causal_mask(
            config=lm.config,
            input_embeds=inputs_embeds,
            attention_mask=vl_input["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=vl_input["text_position_ids"],
        )

        # 4. RoPE embeddings
        position_embeddings = lm.rotary_emb(inputs_embeds, vl_input["position_ids"])

        # 5. Scatter image embeddings into text embedding
        idx = vl_input["visual_indices"].unsqueeze(0).unsqueeze(-1).expand(1, -1, inputs_embeds.shape[-1])
        inputs_embeds = inputs_embeds.scatter(1, idx, image_embeds.unsqueeze(0))

        return {
            "inputs_embeds": inputs_embeds,
            "causal_mask": causal_mask,
            "text_position_ids": vl_input["text_position_ids"],
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "deepstack_visual_embeds": deepstack_image_embeds,
            "visual_indices": vl_input["visual_indices"],
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
                hidden_states = hidden_states.scatter_add(1, idx, visual_embed.unsqueeze(0))

        # Return pre-norm hidden states (no final norm)
        return hidden_states

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        _prof = getattr(self, '_enable_profiling', False)
        _sync = getattr(self, '_profile_sync', False) and _prof

        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}

        # Step 0: Ensure visual cache (eager, not compilable)
        self._ensure_visual_cache()

        # Step 1: Pre-compute non-compilable values
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        visual_indices = image_mask[0].nonzero().squeeze(-1)
        vl_input["visual_indices"] = visual_indices
        vl_input["image_mask"] = image_mask

        # Step 1b: Position IDs (get_rope_index uses .tolist(), not compilable)
        if _prof:
            if _sync: torch.npu.synchronize()
            t0 = time.time()
        qwen3vl_model = self.model.model
        position_ids, _ = qwen3vl_model.get_rope_index(
            vl_input["input_ids"],
            image_grid_thw=vl_input["image_grid_thw"],
            attention_mask=vl_input["attention_mask"],
        )
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        vl_input["position_ids"] = position_ids
        vl_input["text_position_ids"] = text_position_ids
        if _prof:
            if _sync: torch.npu.synchronize()
            t_rope = time.time() - t0

        # Step 2: Preprocess (compilable with torchair)
        if _prof:
            if _sync: torch.npu.synchronize()
            t0 = time.time()
        lm_kwargs = self._preprocess_vl_input(vl_input)
        if _prof:
            if _sync: torch.npu.synchronize()
            t_preprocess = time.time() - t0

        # Step 3: Language model (compilable with torchair)
        if _prof:
            if _sync: torch.npu.synchronize()
            t0 = time.time()
        hidden_states = self._language_model_forward(**lm_kwargs)
        if _prof:
            if _sync: torch.npu.synchronize()
            t_lm = time.time() - t0

        # Step 4: Output processing
        attention_mask = vl_input["attention_mask"] == 1

        if _prof:
            self._prof_step = getattr(self, '_prof_step', 0) + 1
            if self._prof_step <= 4:
                print(f"[PROF] backbone: rope_idx={t_rope*1000:.1f}ms  "
                      f"preprocess={t_preprocess*1000:.1f}ms  lm={t_lm*1000:.1f}ms  "
                      f"total={((t_rope+t_preprocess+t_lm)*1000):.1f}ms")

        return BatchFeature(
            data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )
