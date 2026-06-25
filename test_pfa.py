"""PFA (npu_prompt_flash_attention) vs explicit matmul+softmax on 310P.

Tests:
  1. PFA callable on 310P
  2. Numerical diff vs reference matmul+softmax
  3. Latency: PFA vs matmul+softmax

Run:
  python test_pfa.py
"""

import math
import time

import torch
import torch_npu


def reference_attention(q, k, v, scale):
    """fp32-softmax reference, mirrors current visual attention impl."""
    aw = torch.matmul(q, k.transpose(-2, -1)) * scale
    aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def pfa_attention(q, k, v, num_heads, scale):
    return torch_npu.npu_prompt_flash_attention(
        q, k, v,
        num_heads=num_heads,
        input_layout="BNSD",
        scale_value=scale,
    )


def bench(fn, n=100):
    """Warmup + timed loop. Returns avg ms."""
    for _ in range(5):  # warmup
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / n * 1000


def run_case(name, B, N, S, D):
    print(f"\n{'='*70}")
    print(f"CASE: {name}  shape=[B={B}, N={N}, S={S}, D={D}] (BNSD)")
    print('='*70)

    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    v = torch.randn(B, N, S, D, dtype=torch.float16).npu()

    # 1. PFA callable?
    try:
        out_pfa = pfa_attention(q, k, v, N, scale)
        print(f"[OK] PFA called, output shape={tuple(out_pfa.shape)}, dtype={out_pfa.dtype}")
    except Exception as e:
        print(f"[FAIL] PFA not callable on this device: {e}")
        return False

    # 2. Numerical diff
    with torch.no_grad():
        out_ref = reference_attention(q, k, v, scale)
        diff = (out_pfa.float() - out_ref.float()).abs()
        rel = diff.mean().item() / (out_ref.float().abs().mean().item() + 1e-12)
        print(f"[diff]   max={diff.max().item():.6f}  mean={diff.mean().item():.6f}  rel_mean={rel*100:.4f}%")

    # 3. Latency
    t_pfa = bench(lambda: pfa_attention(q, k, v, N, scale))
    t_ref = bench(lambda: reference_attention(q, k, v, scale))
    speedup = t_ref / t_pfa if t_pfa > 0 else 0
    print(f"[time]   PFA={t_pfa:.3f} ms   matmul+softmax={t_ref:.3f} ms   speedup={speedup:.2f}x")

    return True


def main():
    print(">>> PFA vs matmul+softmax on Ascend NPU")
    print(f">>> torch_npu version: {torch_npu.__version__}")
    print(f">>> device: {torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A'}")

    # Case 1: 我们的 visual attention 实际 shape
    ok = run_case("visual attention (ours)", B=4, N=16, S=256, D=128)
    if not ok:
        print("\n>>> PFA unusable on this device. Aborting.")
        return

    # Case 2: 不同 head 数（验证一般性）
    run_case("more heads",          B=4, N=32, S=256, D=128)

    # Case 3: 不同 seq len
    run_case("longer seq",          B=4, N=16, S=512, D=128)

    # Case 4: 单 batch（最小开销场景）
    run_case("single batch",        B=1, N=16, S=256, D=128)


if __name__ == "__main__":
    main()
