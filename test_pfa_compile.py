"""Can PFA + causal mask be compiled by torchair on 310P1?

Tests 3 scenarios:
  1. PFA + atten_mask in a Model.forward(), torch.compile (torchair backend)
  2. PFA without mask (visual-encoder-like case)
  3. PFA + repeat_kv (LM GQA case)

If (1) fails to compile or silently falls back, that explains why the LM patch
didn't speed up.
"""

import math
import time

import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig


def make_backend():
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    return tng.get_npu_backend(compiler_config=config)


def repeat_kv(t, n_rep):
    if n_rep == 1:
        return t
    B, NKV, S, D = t.shape
    t = t[:, :, None, :, :].expand(B, NKV, n_rep, S, D)
    return t.reshape(B, NKV * n_rep, S, D)


def test_compile(name, fn, args):
    """Compile fn with torchair, run, time it. Returns (compiled_ok, ms)."""
    print(f"\n--- {name} ---")
    backend = make_backend()
    try:
        compiled = torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)
    except Exception as e:
        print(f"  compile setup FAIL: {str(e)[:200]}")
        return False, 0

    # Warmup + time
    try:
        for _ in range(3):
            _ = compiled(*args)
        torch.npu.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = compiled(*args)
        torch.npu.synchronize()
        ms = (time.time() - t0) / 20 * 1000
        print(f"  compiled OK, avg {ms:.3f} ms / call")
        return True, ms
    except Exception as e:
        msg = str(e).replace('\n', ' ')
        print(f"  compile FAIL at runtime: {msg[:300]}")
        return False, 0


def main():
    print(f">>> device: {torch.npu.get_device_name(0)}")

    # Fixed shapes
    B, N, S, D = 1, 16, 288, 128  # S=288 = 16-aligned
    KV_N = 8  # GQA
    scale = 1.0 / math.sqrt(D)
    mask = (~torch.tril(torch.ones(S, S, dtype=torch.bool))).npu()

    # Pre-built inputs
    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k_mha = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    v_mha = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k_gqa = torch.randn(B, KV_N, S, D, dtype=torch.float16).npu()
    v_gqa = torch.randn(B, KV_N, S, D, dtype=torch.float16).npu()

    # Test 1: PFA with mask (LM-like, no GQA)
    class M1(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=N, input_layout="BNSD",
                scale_value=scale, atten_mask=mask, sparse_mode=0,
            )
    ok1, ms1 = test_compile(
        "PFA + mask (no GQA)",
        M1(), (q, k_mha, v_mha, mask)
    )

    # Test 2: PFA without mask (visual-encoder-like)
    class M2(torch.nn.Module):
        def forward(self, q, k, v):
            return torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=N, input_layout="BNSD",
                scale_value=scale,
            )
    ok2, ms2 = test_compile(
        "PFA no mask",
        M2(), (q, k_mha, v_mha)
    )

    # Test 3: PFA + mask + repeat_kv (full LM scenario)
    class M3(torch.nn.Module):
        def forward(self, q, k_gqa, v_gqa, mask):
            k = repeat_kv(k_gqa, N // KV_N)
            v = repeat_kv(v_gqa, N // KV_N)
            return torch_npu.npu_prompt_flash_attention(
                q, k, v, num_heads=N, input_layout="BNSD",
                scale_value=scale, atten_mask=mask, sparse_mode=0,
            )
    ok3, ms3 = test_compile(
        "PFA + mask + repeat_kv (full LM)",
        M3(), (q, k_gqa, v_gqa, mask)
    )

    # Summary
    print(f"\n>>> Summary:")
    print(f"  PFA + mask:           {'OK' if ok1 else 'FAIL'} ({ms1:.2f}ms)")
    print(f"  PFA no mask:          {'OK' if ok2 else 'FAIL'} ({ms2:.2f}ms)")
    print(f"  PFA + mask + repeat:  {'OK' if ok3 else 'FAIL'} ({ms3:.2f}ms)")

    if ok1 and ok2 and ok3:
        print(f"\n>>> All compile OK. LM patch should work.")
        print(f">>> If LM time doesn't drop in real model, look elsewhere.")
    elif not ok1 and ok2:
        print(f"\n>>> PFA + mask does NOT compile on torchair.")
        print(f">>> This is the smoking gun: explains why LM patch didn't help.")
    elif not ok3:
        print(f"\n>>> Full LM scenario doesn't compile.")
        print(f">>> Maybe repeat_kv pattern, maybe mask, investigate.")


if __name__ == "__main__":
    main()
