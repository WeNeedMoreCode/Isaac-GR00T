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

    def _preprocess_vl_input(self, vl_input: dict) -> dict:
        """Run all data-dependent preprocessing (image encoding, embedding, position_ids).

        Mirrors the first half of Qwen3VLModel.forward so that only the language_model
        call remains for torchair compilation.
        """
        qwen3vl_model = self.model.model  # Qwen3VLModel

        # 1. Text embedding
        inputs_embeds = qwen3vl_model.get_input_embeddings()(vl_input["input_ids"])

        # 2. Image encoding
        pixel_values = vl_input["pixel_values"]
        image_grid_thw = vl_input["image_grid_thw"]
        image_embeds, deepstack_image_embeds = qwen3vl_model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        # 3. Scatter image embeddings into text embedding
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds)

        # 4. Visual position masks and deepstack features
        visual_pos_masks = image_mask  # [B, seq_len]
        deepstack_visual_embeds = deepstack_image_embeds

        # 5. Compute position_ids (data-dependent)
        position_ids, rope_deltas = qwen3vl_model.get_rope_index(
            vl_input["input_ids"],
            image_grid_thw=image_grid_thw,
            attention_mask=vl_input["attention_mask"],
        )

        return {
            "input_ids": None,
            "position_ids": position_ids,
            "attention_mask": vl_input["attention_mask"],
            "past_key_values": None,
            "inputs_embeds": inputs_embeds,
            "cache_position": None,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }

    def _language_model_forward(self, **kwargs) -> torch.Tensor:
        """Run the language model forward pass (compilable with torchair)."""
        outputs = self.model.model.language_model(**kwargs, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
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
