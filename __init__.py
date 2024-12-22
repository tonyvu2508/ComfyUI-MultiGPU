import time
import torch

from copy import deepcopy
import comfy.model_management

current_device = "cuda:0"
def get_torch_device_patched():
    global current_device
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
    ):
        return torch.device("cpu")

    return torch.device(current_device)

comfy.model_management.get_torch_device = get_torch_device_patched

def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = deepcopy(cls.INPUT_TYPES())
            inputs["required"]["device"] = ([f"cuda:{i}" for i in range(torch.cuda.device_count())],)
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device, **kwargs):
            global current_device
            current_device = device
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)
    
    return NodeOverride

# This is to make sure the other nodes are already loaded
time.sleep(2)
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS

TARGET_NODE_NAMES = {
    "UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoader",
    "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF", "TripleCLIPLoaderGGUF", 
}

TARGET_NODE_CLASS_MAPPINGS = {}
for name in TARGET_NODE_NAMES:
    if name not in GLOBAL_NODE_CLASS_MAPPINGS:
        continue
    TARGET_NODE_CLASS_MAPPINGS[name] = GLOBAL_NODE_CLASS_MAPPINGS[name]

NODE_CLASS_MAPPINGS = {}
for name, cls in TARGET_NODE_CLASS_MAPPINGS.items():
    NODE_CLASS_MAPPINGS[f"{name}_MultiGPU"] = override_class(cls)