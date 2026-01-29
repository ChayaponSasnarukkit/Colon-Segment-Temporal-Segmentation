import torch
import os
import sys

print(f"Python: {sys.executable}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

print("\n--- Attempting to load selective_scan_cuda ---")
try:
    import selective_scan_cuda
    print("✅ SUCCESS: selective_scan_cuda loaded directly!")
except Exception as e:
    print(f"❌ FAIL: Could not load selective_scan_cuda.\nERROR: {e}")

print("\n--- Checking Mamba Inner Function ---")
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
    print(f"mamba_inner_fn value: {mamba_inner_fn}")
    if mamba_inner_fn is None:
        print("❌ CRITICAL: mamba_inner_fn is None. The fallback failed.")
    else:
        print("✅ SUCCESS: mamba_ssm function is ready.")
except Exception as e:
    print(f"❌ FAIL: Error importing mamba_ssm ops.\nERROR: {e}")
