"""测试 RC 设备上 torch.empty + 原地随机填充方案。

用法：
  python randn_alternatives_test.py                  # 默认测 normal_（randn 的平替）
  python randn_alternatives_test.py --method normal
  python randn_alternatives_test.py --method uniform
  python randn_alternatives_test.py --method bernoulli

注意：只有 normal_() 跟 torch.randn 同分布（高斯），是真正的平替。
      uniform_ 和 bernoulli_ 分布不同，只是顺便验证 NPU 上能不能跑原地随机算子。
"""
import argparse
import time

import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["normal", "uniform", "bernoulli"],
        default="normal",
        help="randn 平替方法（normal 是真平替）",
    )
    parser.add_argument("--shape", type=int, nargs="+", default=[16, 1024])
    args = parser.parse_args()

    shape = tuple(args.shape)
    device = "npu:0"
    dtype = torch.float16

    print(f"method={args.method}, shape={shape}, device={device}, dtype={dtype}")

    t0 = time.time()
    x = torch.empty(shape, device=device, dtype=dtype)
    if args.method == "normal":
        x.normal_()
    elif args.method == "uniform":
        x.uniform_(-1, 1)
    elif args.method == "bernoulli":
        x.bernoulli_(0.5)
    torch.npu.synchronize()
    elapsed = time.time() - t0

    print(f"✅ OK ({elapsed*1000:.1f}ms)")
    print(f"   shape={tuple(x.shape)}")
    print(f"   mean={x.float().mean().item():.4f}")
    print(f"   std ={x.float().std().item():.4f}")
    print(f"   min ={x.float().min().item():.4f}")
    print(f"   max ={x.float().max().item():.4f}")


if __name__ == "__main__":
    main()
