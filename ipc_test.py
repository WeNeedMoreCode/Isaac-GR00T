"""验证 torch_npu IPC: 父进程加载 NPU 权重，spawn 子进程复用，不重新加载。

判断标准:
- 子进程能拿到 weight，不报 share_memory_ not supported
- 子进程访问时 NPU 内存不翻倍（验证真共享，不是 copy）
- 子进程能用共享权重做计算
- 子进程退出后父进程权重仍可用

注意: RC 设备(310P1)上 torch.randn(..., device='npu') 走 aicpu 会报错，
      所以先在 CPU 生成再 .to(npu)，跟主推理代码里的修复一致。
"""
import os
import time
import torch
import torch_npu
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)  # NPU 不支持 fork


def worker(weight, label):
    """子进程：接收共享权重，做 matmul 验证可用。"""
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    print(f"[child {os.getpid()}] {label} start, "
          f"weight.device={weight.device}, shape={weight.shape}")

    # 关键验证：内存不翻倍
    mem_before = torch.npu.memory_allocated() / 1e9
    # RC 上 randn 走 CPU→device
    x = torch.randn(8, weight.shape[1], dtype=weight.dtype).to(weight.device)
    y = x @ weight.T
    torch.npu.synchronize()
    mem_after = torch.npu.memory_allocated() / 1e9
    print(f"[child {os.getpid()}] matmul done, y.shape={y.shape}, "
          f"mem: before={mem_before:.3f}GB after={mem_after:.3f}GB")


def main():
    device = "npu:0"
    print(f"[parent {os.getpid()}] allocating weight on {device}...")
    t0 = time.time()
    # RC 上 randn 走 CPU→device（跟 gr00t_n1d7.py 里的修复一致）
    weight = torch.randn(4096, 4096, dtype=torch.float16).to(device)
    print(f"[parent] allocated in {time.time()-t0:.3f}s, "
          f"npu mem: {torch.npu.memory_allocated()/1e9:.3f}GB")

    # 关键：标记为可共享
    try:
        weight.share_memory_()
        print(f"[parent] share_memory_() OK")
    except Exception as e:
        print(f"[parent] share_memory_() FAILED: {e}")
        print(f"[parent] IPC not supported in this torch_npu version, abort.")
        return

    # spawn 两个子进程串行跑（模拟编排器场景）
    for i in range(2):
        print(f"\n[parent] spawning child {i}...")
        t0 = time.time()
        p = mp.Process(target=worker, args=(weight, f"child-{i}"))
        p.start()
        p.join()
        print(f"[parent] child {i} exited after {time.time()-t0:.3f}s, "
              f"rc={p.exitcode}")

    # 父进程的权重还能用？
    print(f"\n[parent] re-testing weight after children exited...")
    z = weight @ weight.T
    torch.npu.synchronize()
    print(f"[parent] OK, z.shape={z.shape}, "
          f"npu mem: {torch.npu.memory_allocated()/1e9:.3f}GB")


if __name__ == "__main__":
    main()
