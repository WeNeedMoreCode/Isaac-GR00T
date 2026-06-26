"""Spy on LM attention_mask: is there padding or is it pure causal?

Monkey-patches _preprocess_vl_input to print attention_mask stats on the first
call, then runs the normal inference. Decides whether PFA patch can ignore
attention_mask (pure causal) or must convert it (has padding).

Run with the same args as standalone_inference_script.py:
  python test_attn_mask.py \\
    --model-path ./checkpoints/GR00T-N1.7-3B \\
    --backbone-path ./checkpoints/Cosmos-Reason2-2B \\
    --traj-ids 0 --steps 1 --action-horizon 8 --device npu \\
    --denoising-steps 1 --no-pipeline --no-nz-cast
"""

import torch
import gr00t.model.modules.qwen3_backbone as qwb

_spied = False
_orig = qwb.Qwen3Backbone._preprocess_vl_input


def _spy(self, vl_input):
    global _spied
    if not _spied:
        am = vl_input.get("attention_mask")
        if am is not None:
            print("\n" + "=" * 60)
            print("[SPY] LM attention_mask (from vl_input)")
            print("=" * 60)
            print(f"  shape:        {tuple(am.shape)}")
            print(f"  dtype:        {am.dtype}")
            print(f"  device:       {am.device}")
            print(f"  all 1?        {(am == 1).all().item()}")
            print(f"  has 0?        {(am == 0).any().item()}")
            uniq = am.unique().tolist()
            print(f"  unique vals:  {uniq[:10]}")
            if (am != 1).any():
                pad = (am != 1)
                print(f"  >>> PADDING DETECTED")
                print(f"  >>> total: {pad.sum().item()}, per-batch: {pad.sum(dim=-1).tolist()}")
                print(f"  >>> PFA patch must convert attention_mask to bool mask")
            else:
                print(f"  >>> No padding. PFA patch can use pure causal mask.")
        _spied = True
    return _orig(self, vl_input)


qwb.Qwen3Backbone._preprocess_vl_input = _spy

# Also spy on the cached causal_mask (what's actually passed to LM)
_orig_forward = qwb.Qwen3Backbone.forward
_spied_mask = False


def _spy_forward(self, vl_input):
    global _spied_mask
    result = _orig_forward(self, vl_input)
    if not _spied_mask:
        # vl_input now contains _cached_causal_mask after forward
        cm = vl_input.get("_cached_causal_mask")
        if cm is not None:
            print("\n" + "=" * 60)
            print("[SPY] Cached causal_mask (passed to LM)")
            print("=" * 60)
            print(f"  shape:        {tuple(cm.shape)}")
            print(f"  dtype:        {cm.dtype}")
            print(f"  min:          {cm.min().item():.4f}")
            print(f"  max:          {cm.max().item():.4f}")
            # additive mask: 0 = attend, large negative = mask-out
            # count how many positions are "masked" (very negative)
            masked = (cm < -1e3).sum().item()
            total = cm.numel()
            print(f"  masked cells: {masked} / {total}  ({masked/total*100:.1f}%)")
            # For a pure causal: roughly half cells masked (upper triangle)
            # For S=1200, half = ~720000 / 1440000
            print(f"  (pure causal expects ~50% masked)")
        _spied_mask = True
    return result


qwb.Qwen3Backbone.forward = _spy_forward


# Run normal inference
import tyro
from scripts.deployment.standalone_inference_script import main, ArgsConfig

config = tyro.cli(ArgsConfig)
main(config)
