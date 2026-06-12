"""Profile GR00T N1.7 inference for 3 steps on NPU."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch_npu
from gr00t.policy.gr00t_policy import Gr00tPolicy

# Load policy
policy = Gr00tPolicy(
    "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    "./checkpoints/GR00T-N1.7-3B",
    device="npu:0",
    backbone_path="./checkpoints/Cosmos-Reason2-2B/",
    compile=False,
    nz_cast=False,
)
policy.model.action_head.num_inference_timesteps = 1

# Prepare dummy observation using real data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import VLAStepData, MessageType
from gr00t.data.dataset import LeRobotEpisodeLoader
from copy import deepcopy

loader = LeRobotEpisodeLoader(
    dataset_path="demo_data/droid_sample",
    modality_configs=None,
    video_backend="decord",
)
traj = loader[0]
modality_configs = deepcopy(loader.modality_configs)
modality_configs.pop("action")

# Prepare first step
data_point = traj[0]
obs = {}
for k, v in data_point.states.items():
    obs[f"state.{k}"] = v
for k, v in data_point.images.items():
    obs[f"video.{k}"] = __import__("numpy").array(v)
for language_key in loader.modality_configs["language"].modality_keys:
    obs[language_key] = data_point.text

from gr00t.policy.gr00t_policy import Gr00tPolicy as _GP
messages = [{"type": MessageType.EPISODE_STEP.value, "content": VLAStepData(images=data_point.images, states=data_point.states, actions={}, language=data_point.text)}]
parsed_inputs = [policy.processor(messages)]
collated_inputs = policy.collate_fn(parsed_inputs)
import torch as _t
collated_inputs = {k: v.to(dtype=_t.float16) if isinstance(v, _t.Tensor) else v for k, v in collated_inputs.items()}

# Profile 3 steps: warmup=1, active=2
print("Starting profiling (1 warmup + 2 active = 3 steps)...")

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_result"),
    record_shapes=True,
    with_stack=True,
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        data_simplification=True,
    ),
) as prof:
    for step in range(3):
        print(f"  Step {step+1}/3...")
        model_pred = policy.dispatch_inference(collated_inputs)
        action_chunk, _ = policy.decode_action(model_pred, [data_point.states])
        prof.step()

print("Profiling done. Results saved to ./prof_result/")
print("View with: mindstudio-insight ./prof_result/")
