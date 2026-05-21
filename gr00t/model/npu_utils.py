# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU utilities for GR00T N1.7 inference on Ascend.

This module provides helpers adopted from the pi0.5 torchair adaptation pattern:
1. NPU rotary position embedding (RoPE) replacement
2. FRACTAL_NZ weight format casting for Linear layers
3. torchair graph compilation helpers
"""

import importlib
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoPE replacement
# ---------------------------------------------------------------------------

def apply_npu_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU-compatible rotary position embedding using torch_npu.npu_rotary_mul.

    Mirrors the pi0.5 adaptation pattern:
    https://gitcode.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/embodied_ai/vla/pi05
    """
    import torch_npu

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    n_q = q.shape[1]
    n_k = k.shape[1]
    merged_states = torch.cat([q, k], dim=1)
    merged_rot = torch_npu.npu_rotary_mul(merged_states, cos, sin)
    q_embed, k_embed = merged_rot.split([n_q, n_k], dim=1)
    return q_embed, k_embed


def patch_qwen3_rope_for_npu():
    """Monkey-patch Qwen3 RoPE to use NPU kernel.

    Tries multiple known module paths since the exact package layout varies
    across transformers versions.
    """
    candidates = [
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "apply_rotary_pos_emb"):
                mod.apply_rotary_pos_emb = apply_npu_rope
                logger.info(f"Patched RoPE in {mod_name}")
                return
        except Exception:
            continue
    logger.warning(
        "Could not patch Qwen3 RoPE for NPU. "
        "If you encounter errors during backbone forward, check the transformers version."
    )


# ---------------------------------------------------------------------------
# FRACTAL_NZ weight casting
# ---------------------------------------------------------------------------

def format_cast_to_nz(model: nn.Module) -> nn.Module:
    """Cast all Linear layer weights to FRACTAL_NZ format for NPU.

    Adopted from the pi0.5 torchair adaptation.
    """
    import torch_npu

    FRACTAL_NZ = 29
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                weight_nz = torch_npu.npu_format_cast(module.weight, FRACTAL_NZ)
                module.weight = nn.Parameter(weight_nz, requires_grad=False)
                logger.debug(f"Converted {name} -> FRACTAL_NZ")
    return model


# ---------------------------------------------------------------------------
# torchair compilation
# ---------------------------------------------------------------------------

def get_npu_backend(frozen_parameter: bool = True, tiling_optimize: bool = True):
    """Build a torchair NPU backend for torch.compile.

    Args:
        frozen_parameter: Freeze parameters for compilation (recommended for inference).
        tiling_optimize: Enable tiling full-sink optimization.
    """
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.experimental_config.frozen_parameter = frozen_parameter
    config.experimental_config.tiling_schedule_optimize = tiling_optimize
    return tng.get_npu_backend(compiler_config=config)


def compile_for_npu(model: nn.Module, method_name: str = "forward") -> None:
    """Replace a model method with its torchair-compiled version.

    Example:
        compile_for_npu(backbone, "forward")
        compile_for_npu(action_head.model, "forward")
    """
    npu_backend = get_npu_backend()
    original = getattr(model, method_name)
    compiled = torch.compile(original, dynamic=False, fullgraph=True, backend=npu_backend)
    setattr(model, method_name, compiled)
    logger.info(f"Compiled {model.__class__.__name__}.{method_name} with torchair")
