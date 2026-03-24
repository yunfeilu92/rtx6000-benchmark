"""
Patch PyTorch's pin_memory to handle nuPlan Feature objects.

Feature objects have a .data attribute (dict of tensors) that PyTorch's
default pin_memory function doesn't know about. This patch adds support.

Must be imported BEFORE DataLoader is created.
"""
import copy
import collections.abc
import torch
import torch.utils.data._utils.pin_memory as pm

_original_pin_memory = pm.pin_memory

def patched_pin_memory(data, device=None):
    """Extended pin_memory that handles objects with .data dict attribute."""
    # Handle custom Feature objects (PlutoFeature, Trajectory, etc.)
    if hasattr(data, 'data') and not isinstance(data, torch.Tensor):
        if isinstance(data.data, dict):
            data.data = {k: patched_pin_memory(v, device) for k, v in data.data.items()}
            return data
        elif isinstance(data.data, torch.Tensor):
            data.data = data.data.pin_memory(device)
            return data
    # Fall back to original for everything else
    return _original_pin_memory(data, device)

# Apply patch
pm.pin_memory = patched_pin_memory
print("[PATCH] pin_memory patched to handle Feature objects")
