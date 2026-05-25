"""Verify two NPU dtype/device transfer behaviors.

1. model.to(device, dtype) vs model.to(device) + model.half()
2. conv.half().to(npu) vs conv.to(npu).half() weight dtype
"""
import torch
import torch_npu

device = "npu:0"

print("=" * 60)
print("Test 1: model.to(device, dtype)")
print("=" * 60)

model = torch.nn.Linear(10, 10)
try:
    model = model.to(device=device, dtype=torch.float16)
    print(f"  to(device, dtype) -> weight dtype: {model.weight.dtype}")
    print("  Result: SUCCEEDED (unexpected if our assumption holds)")
except Exception as e:
    print(f"  to(device, dtype) -> FAILED: {e}")
    print("  Result: FAILED (matches our assumption)")

model2 = torch.nn.Linear(10, 10)
model2 = model2.to(device=device)
model2 = model2.half()
print(f"  to(device) + half() -> weight dtype: {model2.weight.dtype}")

print()
print("=" * 60)
print("Test 2: Conv3d half/npu order")
print("=" * 60)

conv1 = torch.nn.Conv3d(3, 16, kernel_size=3)
conv1 = conv1.half()
print(f"  After half(): weight dtype = {conv1.weight.dtype}")
conv1 = conv1.to(device)
print(f"  After to(npu): weight dtype = {conv1.weight.dtype}")
print(f"  Result: {'RESET to float32' if conv1.weight.dtype == torch.float32 else 'STAYED float16'}")

conv2 = torch.nn.Conv3d(3, 16, kernel_size=3)
conv2 = conv2.to(device)
print(f"  After to(npu): weight dtype = {conv2.weight.dtype}")
conv2 = conv2.half()
print(f"  After half(): weight dtype = {conv2.weight.dtype}")
print(f"  Result: {'RESET to float32' if conv2.weight.dtype == torch.float32 else 'STAYED float16'}")
