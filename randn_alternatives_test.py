"""测试 RC 设备上 torch.randn 各种平替方案。

每个方案尝试一次，报告：能跑 / 挂 / 数值是否正常。
"""
import time
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
device = "npu:0"
dtype = torch.float16
shape = (16, 1024)  # 类似 GR00T 推理时的 noise 形状


def _stats(name, x):
    """打印张量统计，看是不是真随机。"""
    if isinstance(x, torch.Tensor):
        print(f"  {name}: shape={tuple(x.shape)} "
              f"mean={x.float().mean().item():.4f} "
              f"std={x.float().std().item():.4f} "
              f"min={x.float().min().item():.4f} "
              f"max={x.float().max().item():.4f}")
    else:
        print(f"  {name}: {x}")


def test_method(name, fn):
    """跑一个方案，捕获异常，打印结果。"""
    print(f"\n=== {name} ===")
    torch.npu.synchronize()
    t0 = time.time()
    try:
        x = fn()
        torch.npu.synchronize()
        elapsed = time.time() - t0
        print(f"  ✅ OK ({elapsed*1000:.1f}ms)")
        _stats("result", x)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        err = str(e)
        if len(err) > 200:
            err = err[:200] + "..."
        print(f"  ❌ FAILED ({elapsed*1000:.1f}ms): {err}")
        return False


def main():
    print(f"device={device}, dtype={dtype}, shape={shape}")
    print(f"torch_npu version: {torch_npu.__version__}")

    results = {}

    # 方案 0：直接 randn（baseline，预期失败）
    results["randn_direct"] = test_method(
        "0. torch.randn(direct on NPU)",
        lambda: torch.randn(shape, device=device, dtype=dtype),
    )

    # 方案 1：CPU 生成 + .to(npu)（已知能用）
    results["cpu_then_to"] = test_method(
        "1. torch.randn(CPU) + .to(npu)",
        lambda: torch.randn(shape, dtype=dtype).to(device),
    )

    # 方案 2a：empty + normal_
    results["empty_normal"] = test_method(
        "2a. torch.empty(npu) + .normal_()",
        lambda: torch.empty(shape, device=device, dtype=dtype).normal_(),
    )

    # 方案 2b：empty + uniform_
    results["empty_uniform"] = test_method(
        "2b. torch.empty(npu) + .uniform_(-1, 1)",
        lambda: torch.empty(shape, device=device, dtype=dtype).uniform_(-1, 1),
    )

    # 方案 2c：empty + bernoulli_
    results["empty_bernoulli"] = test_method(
        "2c. torch.empty(npu) + .bernoulli_(0.5)",
        lambda: torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5),
    )

    # 方案 3：NPU empty + fill_(0) + 正态化（手工实现 Box-Muller 太麻烦，跳过）

    # 方案 4：empty + copy_ from CPU randn（同方案 1，但显式）
    def cpu_copy():
        x = torch.empty(shape, device=device, dtype=dtype)
        cpu_x = torch.randn(shape, dtype=dtype)
        x.copy_(cpu_x)
        return x

    results["empty_copy_cpu"] = test_method(
        "4. torch.empty(npu).copy_(torch.randn(cpu))",
        cpu_copy,
    )

    # 方案 5：torch_npu 自己的随机数 API（如果有）
    npu_random_fns = [
        attr for attr in dir(torch_npu.npu)
        if "random" in attr.lower() or "randn" in attr.lower()
    ]
    print(f"\n[torch_npu.npu random-related attrs]: {npu_random_fns}")

    # 总结
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name}")


if __name__ == "__main__":
    main()
