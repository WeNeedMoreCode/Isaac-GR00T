"""Diagnose: which mask shape/dtype (if any) works with PFA on 310P1?

The original test_pfa_causal.py hit aicore exception with 2D bool mask.
This script enumerates mask variants to find any working combo, or confirm
that 310P1 PFA simply doesn't support atten_mask.
"""

import math
import torch
import torch_npu


def reference_causal_attention(q, k, v, scale):
    B, N, S, D = q.shape
    aw = torch.matmul(q, k.transpose(-2, -1)) * scale
    additive = torch.zeros(S, S, dtype=aw.dtype, device=aw.device)
    tri = torch.triu_indices(S, S, offset=1)
    additive[tri[0], tri[1]] = torch.finfo(aw.dtype).min
    aw = aw + additive
    aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def call_pfa(q, k, v, num_heads, scale, mask):
    return torch_npu.npu_prompt_flash_attention(
        q, k, v,
        num_heads=num_heads,
        input_layout="BNSD",
        scale_value=scale,
        atten_mask=mask,
        sparse_mode=0,
    )


def try_mask(tag, mask_fn, S):
    B, N, D = 1, 8, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    v = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    with torch.no_grad():
        ref = reference_causal_attention(q, k, v, scale)

    try:
        mask = mask_fn(S).npu()
    except Exception as e:
        print(f"  [{tag:35s}]  mask construct FAIL: {e}")
        return

    try:
        with torch.no_grad():
            out = call_pfa(q, k, v, N, scale, mask)
        diff = (out.float() - ref.float()).abs().max().item()
        tag_str = "  <-- MATCH" if diff < 0.01 else ""
        print(f"  [{tag:35s}]  OK, max_diff={diff:.6f}{tag_str}")
    except Exception as e:
        msg = str(e).replace('\n', ' ')
        if len(msg) > 100:
            msg = msg[:100] + "..."
        print(f"  [{tag:35s}]  FAIL: {msg}")


def main():
    print(">>> PFA atten_mask diagnostic on 310P1")
    print(f">>> device: {torch.npu.get_device_name(0)}")

    # Try S=32 (slightly larger, avoid edge-case tiling)
    S = 32
    print(f"\n--- S={S} ---")

    variants = [
        # 2D shape
        ("2D [S,S] bool lower=True",        lambda s: torch.tril(torch.ones(s, s, dtype=torch.bool))),
        ("2D [S,S] bool upper=True",        lambda s: ~torch.tril(torch.ones(s, s, dtype=torch.bool))),
        ("2D [S,S] int8 lower=1",           lambda s: torch.tril(torch.ones(s, s, dtype=torch.int8))),
        ("2D [S,S] uint8 lower=1",          lambda s: torch.tril(torch.ones(s, s, dtype=torch.uint8))),
        ("2D [S,S] int8 lower=0 upper=-inf",lambda s: torch.where(torch.tril(torch.ones(s, s)).bool(),
                                                                   torch.zeros(s, s, dtype=torch.int8),
                                                                   torch.full((s, s), -128, dtype=torch.int8))),
        # 3D shape
        ("3D [1,S,S] bool lower=True",      lambda s: torch.tril(torch.ones(1, s, s, dtype=torch.bool))),
        # 4D shape
        ("4D [1,1,S,S] bool lower=True",    lambda s: torch.tril(torch.ones(1, 1, s, s, dtype=torch.bool))),
        ("4D [1,1,S,S] int8 lower=1",       lambda s: torch.tril(torch.ones(1, 1, s, s, dtype=torch.int8))),
        ("4D [1,1,S,S] uint8 lower=1",      lambda s: torch.tril(torch.ones(1, 1, s, s, dtype=torch.uint8))),
    ]

    for tag, fn in variants:
        try_mask(tag, fn, S)

    print("\n--- S=256 (realistic LM seq len) ---")
    # If anything worked above, try realistic S
    for tag, fn in variants[:3]:  # just first 3 to save time
        try_mask(tag, fn, 256)


if __name__ == "__main__":
    main()
