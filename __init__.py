import time
import copy
import torch
import comfy.model_management

current_device = comfy.model_management.get_torch_device()

def get_torch_device_patched():
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
            inputs = copy.deepcopy(cls.INPUT_TYPES())
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

time.sleep(2) # This is to make sure the other nodes are already loaded
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS

TARGET_NODE_NAMES = {
    "UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader",            # ComfyUI Core Nodes - https://github.com/comfyanonymous/ComfyUI
    "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF", "TripleCLIPLoaderGGUF",                             # ComfyUI-GGUF - https://github.com/city96/ComfyUI-GGUF
    "LoadFluxControlNet",                                                                                                                   # x-flux-comfyui - https://github.com/XLabs-AI/x-flux-comfyui
    "Florence2ModelLoader", "DownloadAndLoadFlorence2Model",                                                                                # ComfyUI-Florence2 - https://github.com/kijai/ComfyUI-Florence2
    "LTXVLoader",                                                                                                                           # ComfyUI-LTXVideo - https://github.com/Lightricks/ComfyUI-LTXVideo
}

NODE_CLASS_MAPPINGS = {}
for name in TARGET_NODE_NAMES:
    if name not in GLOBAL_NODE_CLASS_MAPPINGS:
        continue
    NODE_CLASS_MAPPINGS[f"{name}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[name])
