"""测试 RC 设备上 torch.empty + 原地随机填充方案。

每个子进程独立跑，避免失败时污染 NPU context。
"""
import subprocess
import sys
import time

METHODS = [
    ("2a. empty + normal_()", """
import torch, torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
x = torch.empty((16, 1024), device='npu:0', dtype=torch.float16).normal_()
torch.npu.synchronize()
print(f'shape={tuple(x.shape)} mean={x.float().mean().item():.4f} std={x.float().std().item():.4f}')
"""),

    ("2b. empty + uniform_(-1, 1)", """
import torch, torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
x = torch.empty((16, 1024), device='npu:0', dtype=torch.float16).uniform_(-1, 1)
torch.npu.synchronize()
print(f'shape={tuple(x.shape)} mean={x.float().mean().item():.4f} std={x.float().std().item():.4f}')
"""),

    ("2c. empty + bernoulli_(0.5)", """
import torch, torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
x = torch.empty((16, 1024), device='npu:0', dtype=torch.float16).bernoulli_(0.5)
torch.npu.synchronize()
print(f'shape={tuple(x.shape)} mean={x.float().mean().item():.4f} std={x.float().std().item():.4f}')
"""),
]


def run_one(name, code):
    print(f"\n=== {name} ===")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    elapsed = time.time() - t0
    if r.returncode == 0:
        print(f"  ✅ OK ({elapsed*1000:.1f}ms)")
        for line in r.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print(f"  ❌ FAILED ({elapsed*1000:.1f}ms, exit={r.returncode})")
        for line in r.stderr.strip().splitlines()[-10:]:
            print(f"  {line}")
    return r.returncode == 0


def main():
    print("=" * 60)
    print("empty + in-place random test (subprocess isolated)")
    print("=" * 60)

    results = {}
    for name, code in METHODS:
        results[name] = run_one(name, code)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name}")


if __name__ == "__main__":
    main()
