import time
import copy
import torch
import comfy.model_management
import logging
# Add these two new imports:
import os
import importlib.util

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("DEBUG: MultiGPU node initialization started.")

def import_nf4_classes():
    try:
        nf4_path = os.path.join("custom_nodes", "ComfyUI_bnb_nf4_fp4_Loaders", "__init__.py")
        logging.info(f"DEBUG: Attempting to pre-load NF4 classes from {nf4_path}")
        
        if not os.path.exists(nf4_path):
            logging.info("DEBUG: NF4 loader path does not exist")
            return None, None
            
        spec = importlib.util.spec_from_file_location("nf4_loaders", nf4_path)
        nf4_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nf4_module)
        logging.info("DEBUG: Successfully pre-loaded NF4 classes")
        return nf4_module.CheckpointLoaderNF4, nf4_module.UNETLoaderNF4
    except Exception as e:
        logging.info(f"DEBUG: Failed to pre-load NF4 classes: {str(e)}")
        return None, None


# Try to get NF4 classes early
logging.info("DEBUG: Attempting early NF4 class import")
CheckpointLoaderNF4, UNETLoaderNF4 = import_nf4_classes()
logging.info(f"DEBUG: Early NF4 import result - CheckpointLoader: {'Found' if CheckpointLoaderNF4 else 'Not Found'}, UNETLoader: {'Found' if UNETLoaderNF4 else 'Not Found'}")

current_device = comfy.model_management.get_torch_device()
logging.info(f"DEBUG: Initial device: {current_device}")

def get_torch_device_patched():
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
        or "cpu" in str(current_device).lower()
    ):
        logging.info("DEBUG: get_torch_device_patched returning CPU.")
        return torch.device("cpu")
    logging.info(f"DEBUG: get_torch_device_patched returning current_device: {current_device}")
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
            logging.info(f"DEBUG: Overriding node '{cls.__name__}' with device: {device}")
            global current_device
            logging.info(f"DEBUG: Current device before override: {current_device}")
            current_device = device
            logging.info(f"DEBUG: Current device after override: {current_device}")
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)

    return NodeOverride

logging.info("DEBUG: Sleeping for 20 seconds to allow other nodes to load.")
time.sleep(20) # This is to make sure the other nodes are already loaded
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
logging.info("DEBUG: GLOBAL_NODE_CLASS_MAPPINGS after sleep:")
for key in GLOBAL_NODE_CLASS_MAPPINGS:
    logging.info(f"DEBUG:   {key}")

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
    logging.info(f"DEBUG: Checking for '{name}' in GLOBAL_NODE_CLASS_MAPPINGS...")
    if name in GLOBAL_NODE_CLASS_MAPPINGS:
        original_class = GLOBAL_NODE_CLASS_MAPPINGS[name]
        NODE_CLASS_MAPPINGS[f"{name}MultiGPU"] = override_class(original_class)
        logging.info(f"DEBUG: Successfully created MultiGPU version for '{name}'.")
    else:
        logging.warning(f"DEBUG: Node '{name}' not found in GLOBAL_NODE_CLASS_MAPPINGS.")

# Add the NF4 nodes if they were loaded
if CheckpointLoaderNF4 is not None:
    logging.info("DEBUG: Adding pre-loaded CheckpointLoaderNF4 to MultiGPU nodes")
    NODE_CLASS_MAPPINGS["CheckpointLoaderNF4MultiGPU"] = override_class(CheckpointLoaderNF4)

if UNETLoaderNF4 is not None:
    logging.info("DEBUG: Adding pre-loaded UNETLoaderNF4 to MultiGPU nodes")
    NODE_CLASS_MAPPINGS["UNETLoaderNF4MultiGPU"] = override_class(UNETLoaderNF4)

logging.info("DEBUG: MultiGPU node initialization completed.")