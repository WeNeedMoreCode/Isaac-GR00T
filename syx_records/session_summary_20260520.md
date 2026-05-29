# Session Summary — Isaac GR00T N1.7 环境验证与 NPU 适配准备

## 日期
2026-05-20

## 目标
验证 Isaac GR00T N1.7 在远程 GPU 环境的推理能力，为 NPU 适配建立基准。

---

## 已完成工作

### 1. 代码修改与回退

- 初始尝试应用 `gr00t.diff`，因 diff 文件损坏（gr00t_policy.py 部分有重复代码），改为手动修改。
- 最终确认 N1.7 代码应保留原始状态，**`@dataclass` 必须保留**（与 PretrainedConfig 的继承是正常的，之前报错是因为误用了 N1.6 模型）。
- 唯一保留的修改：`scripts/deployment/dgpu/install_deps.sh` 中注释了 `uv sync`（用户环境用 conda，不用 uv）。

### 2. 分支管理

| 分支 | 说明 |
|---|---|
| `main` | 原始分支，超前 origin 2 个提交 |
| `gr00t-fix` | 早期修改分支，已废弃 |
| `n1.7-release` | 基于官仓 N1.7 Release (`23ace64`)，包含 install_deps.sh 修改 |
| `npu_adapt` | **当前工作分支**，基于 `n1.7-release`，用于 NPU 适配 |

### 3. 模型下载

- HuggingFace 直接下载速度极慢（未认证，~458kB/s），经常卡住不动。
- 改用 `huggingface-cli download nvidia/GR00T-N1.7-3B --local-dir ./GR00T-N1.7-3B` 手动下载成功。
- 模型路径约 6.9GB（2 个 safetensors 分片）。

### 4. 推理验证

**环境：**
- 远程 Docker 容器，conda 环境 `gr00t`
- GPU 显存被之前卡死进程占满，先 `kill -9` 清理
- 指定 CUDA 设备：`CUDA_VISIBLE_DEVICES=1`

**Standalone Inference（`standalone_inference_script.py`）：**
```bash
python -m scripts.deployment.standalone_inference_script \
    --model-path ./GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 0 1 \
    --inference-mode pytorch \
    --action-horizon 8
```

结果（与 README benchmark 对齐）：
- Average MSE: **0.008555**
- Average MAE: **0.055992**
- Inference per step: **~259ms**

**Open-Loop Evaluation（`open_loop_eval.py`）：**
- 需先启动 `run_gr00t_server.py`（server），再运行 `open_loop_eval.py`（client）。
- 参数与 standalone 一致时，结果相同（MSE 0.00792, MAE 0.05390），验证 server-client 架构无额外误差。
- 生成可视化图表：`/tmp/open_loop_eval/traj_{id}.jpeg`

### 5. 踩坑记录

| 问题 | 原因 | 解决 |
|---|---|---|
| `non-default argument follows default argument` | 误用 N1.6 模型配 N1.7 代码 | 改用 N1.7 模型 |
| `KeyError: 'Gr00tN1d6'` | 同上 | 同上 |
| `AttributeError: __dataclass_fields__` | 注释了 `@dataclass` | 恢复 `@dataclass`，改用 N1.7 模型 |
| 下载卡住 | HuggingFace 未认证限速 | `huggingface-cli download` 手动下载 |
| CUDA OOM | 之前卡死进程占用 18GB 显存 | `kill -9` 清理旧进程 |
| git-lfs 文件标记 deleted | 本地未下载 LFS 对象 | `GIT_LFS_SKIP_SMUDGE=1 git checkout` 恢复指针文件 |

### 6. 项目架构理解

- **Inference**（`standalone_inference_script.py`）：独立脚本，直接输出指标。
- **Open-Loop**：server-client 架构，动作不执行，只对比预测 vs ground truth。
- **Closed-Loop**：动作真正执行到环境/机器人，测实际表现。
- **Benchmark**：在标准任务（LIBERO、SimplerEnv）上跑 closed-loop，算成功率。
- **EmbodimentTag**：决定 modality config，pretrain tags（zero-shot）vs posttrain tags（需 finetuned checkpoint）。

---

## 当前状态

- 本地分支：`npu_adapt`（基于 `n1.7-release`）
- 远程模型：`./GR00T-N1.7-3B`（6.9GB）
- 远程数据集：`demo_data/droid_sample`（3 episodes，git-lfs 完整）
- GPU 推理验证通过，结果与官方 benchmark 一致。

---

---

---

## NPU 适配进展（2026-05-22 ~ 2026-05-25）

### 核心结论

**NPU standalone inference 已成功跑通**，精度与 GPU 对齐（MSE 0.009 vs 0.0085）。
速度方面：未开启 torchair 时 ~36s/step（比 GPU 慢 140 倍），主要瓶颈在 visual backbone 的 CPU fallback 和 eager 模式未编译。

### 关键 bug 根因：torch_npu `.npu()` 重置 half 权重 + `jit_compile=False` 缺失 Conv3D kernel

1. **`model.to(device, dtype)` 必须拆分**：torch_npu 的 `.to()` 同时传 device+dtype 会报 `Torch not compiled with CUDA enabled`。
   - 解决：`model.to(device)` + `model.half()` 两步。
2. **`.npu()` 会把 half 的 Conv3d 权重重置回 float32**：独立最小测试证实 `conv.half().npu()` 后 weight dtype 变回 float32；`conv.npu().half()` 才是正确的。
3. **`jit_compile=False` 时 Conv3D 算子缺失**：最小测试证实 `nn.Conv3d(3,1024,k=(2,16,16),s=(2,16,16))` 在 `jit_compile=True`（默认）时 OK，在 `False` 时报 `Op Conv3D does not has any binary`。
   - 解决：恢复默认 `jit_compile=True`（注释掉 `set_compile_mode(jit_compile=False)`）。
4. **Qwen3-VL visual backbone 走 NPU 没问题**：在 `jit_compile=True` + `npu()->half()` 正确顺序下，`patch_embed` 可直接在 NPU 上运行，输出 shape `[576, 1024]` 正确。

### 已完成修改（最终版）

| 文件 | 改动 |
|---|---|
| `gr00t/policy/gr00t_policy.py` | float16、`model.to(device)` + `model.half()` 拆分、NPU RoPE patch、torchair 开关 `syx_compile`、FRACTAL_NZ + torchair compile（条件开启） |
| `gr00t/model/modules/qwen3_backbone.py` | eager attention、float16 |
| `gr00t/model/npu_utils.py` | 新增：NPU RoPE、`NpuVisualWrapper`（已注释弃用）、FRACTAL_NZ、torchair backend |
| `scripts/deployment/standalone_inference_script.py` | `--device` 参数、删除 `jit_compile=False`、video_backend 默认改为 `decord` |
| `gr00t/eval/open_loop_eval.py` | `--device` 参数 |
| `gr00t/eval/run_gr00t_server.py` | 删除 `jit_compile=False` |
| `pyproject.toml` | +torch_npu==2.7.1.post4、-CUDA index、torchcodec 0.12.0、triton>=3.5.0、flash-attn 条件依赖、+pytz |

### 性能数据对比

| 指标 | GPU (H100) | NPU (jit_compile=True, 无 torchair) |
|---|---|---|
| MSE | 0.008555 | **0.009359** |
| MAE | 0.055992 | **0.057595** |
| 每步耗时 | ~259ms | **~36s** |

NPU 精度对齐，但速度差距大。下一步开启 torchair 图编译 + 确保 visual 在 NPU 上运行，预期能大幅缩小差距。

### 踩坑记录（新增）

| 问题 | 原因 | 解决 |
|---|---|---|
| torchcodec 0.12.0 依赖 `libnvrtc.so.13` | wheel 编译时链接了 CUDA 13 | 换 backend 为 `decord` |
| decord 编译找不到 `libavfilter` | 缺少 FFmpeg 开发库 | `apt-get install libavfilter-dev` |
| decord 装好后 `VideoReader` 不存在 | `libdecord.so` 没在 Python path 里 | `cp build/libdecord.so` 到 site-packages/decord/ |
| parquet 文件损坏 | git-lfs 未拉取真实文件 | `git lfs pull` |
| `NpuVisualWrapper` 导致 embedding 层 device 错乱 | `nn.Module` 子模块注册 + `.cpu()` side effect | 最终方案：直接删掉 wrapper，改 `jit_compile=True` |
| `Qwen3VLForConditionalGeneration.visual` 是 read-only property | transformers 源码设计 | 实际可写的是 `inner_vl_model.visual` |

### 待办

1. 开启 `syx_compile=1` 测试 torchair 图编译效果
2. 服务化测试（server-client）
3. README 文档和 Claude skill