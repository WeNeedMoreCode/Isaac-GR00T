"""Find PFA + atten_mask S alignment threshold on 310P1.

The real LM has S=277 which triggered:
  attention mask must be NULL, when Qs,Kvs is unAlign

Walk through S values around 277 to find which alignment (16/32/64?) the
310P1 PFA actually requires when atten_mask is provided.

Run:
  python test_pfa_align.py
"""

import math
import torch
import torch_npu


def try_pfa_with_mask(S, D=128, N=16, B=1):
    """Return True if PFA + mask runs without error at this S.

    Note: aicore exceptions on 310P1 may poison the device, making subsequent
    tests fail spuriously. Caller should restart process if None is returned.
    """
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    k = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    v = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    # 2D bool mask, upper-tri True (verified-correct convention)
    mask = (~torch.tril(torch.ones(S, S, dtype=torch.bool))).npu()

    try:
        torch_npu.npu_prompt_flash_attention(
            q, k, v,
            num_heads=N,
            input_layout="BNSD",
            scale_value=scale,
            atten_mask=mask,
            sparse_mode=0,
        )
        torch.npu.synchronize()  # force any async errors to surface
        return True
    except Exception as e:
        msg = str(e)
        # tiling rejection = "must be NULL"
        if 'must be NULL' in msg or 'unAlign' in msg:
            return False
        # aicore exception - device may be poisoned
        return None


def main():
    print(f">>> device: {torch.npu.get_device_name(0)}")
    print(f">>> Finding S alignment threshold for PFA + atten_mask\n")

    # Test S values: aligned + unaligned around 277
    # NOTE: S<32 hits a separate aicore bug (poisons device), skip them.
    test_S = [
        # 32-aligned baseline (known-good)
        256, 272, 288, 304, 320,
        # non-aligned around 277 (real LM S)
        264, 266, 268, 270, 274, 276, 277, 278, 280, 282, 284, 286,
        # 64-aligned controls
        384, 512,
        # boundary near 32
        32, 33, 34, 36, 40, 48, 64, 96,
    ]
    test_S = sorted(set(test_S))

    print(f"{'S':>6}  {'S%16':>5}  {'S%32':>5}  {'S%64':>5}  result")
    print("-" * 50)

    results = {}
    for S in test_S:
        ok = try_pfa_with_mask(S)
        if ok is True:
            tag = "OK"
        elif ok is False:
            tag = "FAIL (mask rejected)"
        else:
            tag = "FAIL (aicore) — device poisoned, stopping"
        results[S] = ok
        print(f"{S:>6}  {S%16:>5}  {S%32:>5}  {S%64:>5}  {tag}")
        if ok is None:
            print(f"\n>>> aicore exception at S={S}. Device is now in bad state.")
            print(f">>> Re-run with this S excluded, or restart process.")
            break

    # Summarize: which alignment seems to be required
    print("\n>>> Summary:")
    ok_16 = [S for S, r in results.items() if r is True and S % 16 == 0]
    ok_32 = [S for S, r in results.items() if r is True and S % 32 == 0]
    ok_64 = [S for S, r in results.items() if r is True and S % 64 == 0]
    fail_unaligned = [S for S, r in results.items() if r is False]

    print(f"  OK with S%16==0:  {[S for S in results if results[S] is True and S%16==0]}")
    print(f"  OK with S%32==0:  {[S for S in results if results[S] is True and S%32==0]}")
    print(f"  FAIL unaligned:   {fail_unaligned}")

    # Try to find smallest alignment that explains all results
    for align in [16, 32, 64, 128]:
        predicts_ok = all((S % align == 0) == (results.get(S) is True) for S in test_S if results.get(S) is not None)
        if predicts_ok:
            print(f"\n  >>> Threshold appears to be: S must be {align}-aligned")
            break
    else:
        print(f"\n  >>> No clean alignment threshold found")


if __name__ == "__main__":
    main()
