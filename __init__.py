import time
import copy
import torch
import comfy.model_management
import os
import importlib.util
import logging

def preload_module_classes(module_path, target_classes):
    try:
        full_path = os.path.join("custom_nodes", module_path, "__init__.py")
        logging.info(f"MultiGPU: Attempting to preload {module_path}")
        if not os.path.exists(full_path):
            logging.info(f"MultiGPU: Module path {module_path} not found")
            return [None] * len(target_classes)
        spec = importlib.util.spec_from_file_location(module_path, full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.info(f"MultiGPU: Successfully loaded {module_path}")
        return [getattr(module, class_name) for class_name in target_classes]
    except Exception as e:
        logging.info(f"MultiGPU: Failed to preload {module_path}: {e}")
        return [None] * len(target_classes)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_device = comfy.model_management.get_torch_device()

def get_torch_device_patched():
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
        or "cpu" in str(current_device).lower()
    ):
        return torch.device("cpu")
    return torch.device(current_device)

comfy.model_management.get_torch_device = get_torch_device_patched

def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            inputs["required"]["device"] = (devices,)
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device, **kwargs):
            global current_device
            current_device = device
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)

    return NodeOverride

# Try to preload NF4 classes
CheckpointLoaderNF4, UNETLoaderNF4 = preload_module_classes(
    "ComfyUI_bnb_nf4_fp4_Loaders", 
    ["CheckpointLoaderNF4", "UNETLoaderNF4"]
)

# Try to preload NF4 classes
logging.info("MultiGPU: Starting NF4 preload")
CheckpointLoaderNF4, UNETLoaderNF4 = preload_module_classes(
    "ComfyUI_bnb_nf4_fp4_Loaders", 
    ["CheckpointLoaderNF4", "UNETLoaderNF4"]
)
logging.info(f"MultiGPU: NF4 preload complete - Checkpoint: {'Found' if CheckpointLoaderNF4 else 'Not Found'}, UNET: {'Found' if UNETLoaderNF4 else 'Not Found'}")

time.sleep(20) # This is to make sure the other nodes are already loaded
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS

TARGET_NODE_NAMES = {
    "UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader",            # ComfyUI Core Nodes - https://github.com/comfyanonymous/ComfyUI
    "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF", "TripleCLIPLoaderGGUF",                             # ComfyUI-GGUF - https://github.com/city96/ComfyUI-GGUF
    "LoadFluxControlNet",                                                                                                                   # x-flux-comfyui - https://github.com/XLabs-AI/x-flux-comfyui
    "Florence2ModelLoader", "DownloadAndLoadFlorence2Model",                                                                                # ComfyUI-Florence2 - https://github.com/kijai/ComfyUI-Florence2
    "LTXVLoader",                                                                                                                           # ComfyUI-LTXVideo - https://github.com/Lightricks/ComfyUI-LTXVideo
    "MMAudioFeatureUtilsLoader", "MMAudioModelLoader", "MMAudioSampler",                                                                    # ComfyUI-MMAudio - https://github.com/kijai/ComfyUI-MMAudio --EXPERIMENTAL--
    "CheckpointLoaderNF4",
}

NODE_CLASS_MAPPINGS = {}
for name in TARGET_NODE_NAMES:
    if name not in GLOBAL_NODE_CLASS_MAPPINGS:
        continue
    NODE_CLASS_MAPPINGS[f"{name}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[name])