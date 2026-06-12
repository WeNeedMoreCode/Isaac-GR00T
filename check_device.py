"""Check which device the backbone and action_head are on after device_map loading."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from gr00t.policy.gr00t_policy import Gr00tPolicy

p = Gr00tPolicy(
    "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    "./checkpoints/GR00T-N1.7-3B",
    device="npu:0",
    backbone_path="./checkpoints/Cosmos-Reason2-2B/",
    compile=False,
    nz_cast=False,
)
print("backbone:", p.model.backbone.model.device)
print("action_head:", next(p.model.action_head.model.parameters()).device)
