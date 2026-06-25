"""PFA + causal mask test for LM attention on 310P.

Three things to verify before patching Qwen3 LM:
  1. Causal mask semantics: bool True=keep or True=mask-out?
  2. GQA workaround: 310P PFA says num_key_value_heads=0 only,
     so we must repeat_kv() to expand GQA to MHA before PFA.
  3. Latency: PFA+causal vs explicit matmul+softmax+mask

Run:
  python test_pfa_causal.py
"""

import math
import time

import torch
import torch_npu


def repeat_kv(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Standard GQA->MHA expansion (mirrors transformers.repeat_kv)."""
    if n_rep == 1:
        return t
    B, NKV, S, D = t.shape
    t = t[:, :, None, :, :].expand(B, NKV, n_rep, S, D)
    return t.reshape(B, NKV * n_rep, S, D)


def reference_causal_attention(q, k, v, scale):
    """Qwen3 eager-style: additive causal mask + fp32 softmax."""
    B, N, S, D = q.shape
    aw = torch.matmul(q, k.transpose(-2, -1)) * scale
    # Additive mask: upper-triangular = finfo.min, lower = 0
    additive = torch.zeros(S, S, dtype=aw.dtype, device=aw.device)
    tri = torch.triu_indices(S, S, offset=1)
    additive[tri[0], tri[1]] = torch.finfo(aw.dtype).min
    aw = aw + additive
    aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def pfa(q, k, v, num_heads, scale, atten_mask):
    return torch_npu.npu_prompt_flash_attention(
        q, k, v,
        num_heads=num_heads,
        input_layout="BNSD",
        scale_value=scale,
        atten_mask=atten_mask,
        sparse_mode=0,
    )


def bench(fn, n=100):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / n * 1000


def test_mask_semantics():
    """Step 1: find which bool mask convention PFA uses."""
    print("\n" + "=" * 70)
    print("STEP 1: Causal mask semantics (which True convention?)")
    print("=" * 70)

    # S=32: S=16 hits a tiling bug in 310P1 PFA. Use S >= 32.
    B, N, S, D = 1, 8, 32, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    v = torch.randn(B, N, S, D, dtype=torch.float16).npu()

    with torch.no_grad():
        ref = reference_causal_attention(q, k, v, scale)

    # 2D bool upper=True is the verified-correct convention from test_pfa_mask_diag.py
    # (True = mask-out, False = keep; upper-triangular True masks the future)
    candidates = [
        ("2D upper=True (~tril) ★", (~torch.tril(torch.ones(S, S, dtype=torch.bool))).npu()),
    ]
    winner = None
    for name, mask in candidates:
        try:
            with torch.no_grad():
                out = pfa(q, k, v, N, scale, mask)
            diff = (out.float() - ref.float()).abs().max().item()
            tag = "  <-- MATCH" if diff < 0.01 else ""
            print(f"  [{name:30s}]  max_diff={diff:.6f}{tag}")
            if diff < 0.01 and winner is None:
                winner = name
        except Exception as e:
            print(f"  [{name:30s}]  FAIL: {e}")

    if winner is None:
        print("\n>>> No mask convention matched! Re-run test_pfa_mask_diag.py to recheck.")
    else:
        print(f"\n>>> Confirmed: {winner}")
    return winner


def _causal_mask(S):
    """Correct causal mask for 310P1 PFA: 2D bool, upper=True (mask-out future)."""
    return (~torch.tril(torch.ones(S, S, dtype=torch.bool))).npu()


def test_gqa_workaround():
    """Step 2: verify repeat_kv() before PFA works for GQA shape."""
    print("\n" + "=" * 70)
    print("STEP 2: GQA -> MHA expansion (repeat_kv then PFA)")
    print("=" * 70)

    # Qwen3-ish GQA: N=16 heads, KV_N=8
    B, N, KV_N, S, D = 1, 16, 8, 256, 128
    n_rep = N // KV_N
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k_gqa = torch.randn(B, KV_N, S, D, dtype=torch.float16).npu()
    v_gqa = torch.randn(B, KV_N, S, D, dtype=torch.float16).npu()

    # Try PFA with GQA shape directly (will likely fail per docs)
    try:
        with torch.no_grad():
            out_direct = pfa(q, k_gqa, v_gqa, num_heads=N, scale=scale, atten_mask=None)
        print(f"  Direct GQA (no repeat):  shape={tuple(out_direct.shape)} -- unexpectedly worked")
    except Exception as e:
        print(f"  Direct GQA (no repeat):  FAIL (as expected): {str(e)[:120]}")

    # Workaround: expand KV to N heads
    k_mha = repeat_kv(k_gqa, n_rep).contiguous()
    v_mha = repeat_kv(v_gqa, n_rep).contiguous()

    # Reference with same expansion
    mask_2d = _causal_mask(S)
    with torch.no_grad():
        ref = reference_causal_attention(q, k_mha, v_mha, scale)

    try:
        with torch.no_grad():
            out = pfa(q, k_mha, v_mha, num_heads=N, scale=scale, atten_mask=mask_2d)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"  After repeat_kv:         shape={tuple(out.shape)}  max_diff={diff:.6f}")
        return diff < 0.05  # tolerate a bit more after expansion
    except Exception as e:
        print(f"  After repeat_kv:         FAIL: {e}")
        return False


def test_performance():
    """Step 3: latency PFA+causal vs explicit, at LM-realistic shape."""
    print("\n" + "=" * 70)
    print("STEP 3: Performance at LM-realistic shape")
    print("=" * 70)

    # Use realistic LM shape (GR00T context ~ 1024 tokens, Qwen3-ish 16 heads)
    cases = [
        ("S=256  (short)",  1, 16, 256,  128),
        ("S=512  (mid)",    1, 16, 512,  128),
        ("S=1024 (long)",   1, 16, 1024, 128),
        ("S=2048 (xlong)",  1, 16, 2048, 128),
    ]

    for name, B, N, S, D in cases:
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
        k = torch.randn(B, N, S, D, dtype=torch.float16).npu()
        v = torch.randn(B, N, S, D, dtype=torch.float16).npu()
        mask = _causal_mask(S)

        try:
            t_pfa = bench(lambda: pfa(q, k, v, N, scale, mask))
        except Exception as e:
            print(f"  {name}:  PFA FAIL: {str(e)[:100]}")
            continue
        t_ref = bench(lambda: reference_causal_attention(q, k, v, scale))
        speedup = t_ref / t_pfa if t_pfa > 0 else 0
        print(f"  {name}:  PFA={t_pfa:.3f} ms   matmul+softmax+mask={t_ref:.3f} ms   speedup={speedup:.2f}x")


def main():
    print(">>> PFA + causal mask feasibility test (for LM attention)")
    print(f">>> torch_npu version: {torch_npu.__version__}")
    print(f">>> device: {torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A'}")

    winner = test_mask_semantics()
    if winner is None:
        print("\n>>> Aborting: can't determine causal mask convention.")
        return

    gqa_ok = test_gqa_workaround()
    if not gqa_ok:
        print("\n>>> GQA workaround did not produce matching output.")
        print(">>> Patching LM still possible but needs more investigation.")

    test_performance()


if __name__ == "__main__":
    main()
