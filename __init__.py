import time
import copy
import torch
import comfy.model_management
import os
import importlib.util
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("MultiGPU: Initialization started")

current_device = comfy.model_management.get_torch_device()
logging.info(f"MultiGPU: Initial device {current_device}")

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

def register_module(module_path, target_nodes):
    try:
        # For core nodes, skip module loading and just register from global mappings
        if not module_path:
            logging.info("MultiGPU: Starting core node registration")
            from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
            for node in target_nodes:
                if node in GLOBAL_NODE_CLASS_MAPPINGS:
                    NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[node])
                    logging.info(f"MultiGPU: Registered core node {node}")
                else:
                    logging.info(f"MultiGPU: Core node {node} not found - this shouldn't happen!")
            return

        # For custom nodes, try to load module first
        full_path = os.path.join("custom_nodes", module_path, "__init__.py")
        logging.info(f"MultiGPU: Checking for module at {full_path}")
        
        if not os.path.exists(full_path):
            logging.info(f"MultiGPU: Module {module_path} not found - skipping")
            return
            
        logging.info(f"MultiGPU: Found {module_path}, attempting to load")
        spec = importlib.util.spec_from_file_location(module_path, full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.info(f"MultiGPU: Executed {module_path} initialization")
        
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
        logging.info(f"MultiGPU: Looking for {module_path} nodes in global mappings")
        for node in target_nodes:
            if node in GLOBAL_NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[node])
                logging.info(f"MultiGPU: Successfully wrapped {node}")
            else:
                logging.info(f"MultiGPU: Node {node} from {module_path} not found in global mappings")
                
    except Exception as e:
        logging.info(f"MultiGPU: Error processing {module_path}: {str(e)}")

NODE_CLASS_MAPPINGS = {}

# Let's test just one new module at a time, starting with GGUF
logging.info("MultiGPU: Starting Core ComfyUI registration")
register_module("", [
    "UNETLoader",
    "VAELoader", 
    "CLIPLoader",
    "DualCLIPLoader", 
    "TripleCLIPLoader", 
    "CheckpointLoaderSimple", 
    "ControlNetLoader"
])

logging.info("MultiGPU: Starting GGUF registration")
register_module("ComfyUI-GGUF", [
    "UnetLoaderGGUF",
    "UnetLoaderGGUFAdvanced",
    "CLIPLoaderGGUF",
    "DualCLIPLoaderGGUF",
    "TripleCLIPLoaderGGUF"
])
logging.info("MultiGPU: Starting X-Flux ControlNet registration")
register_module("x-flux-comfyui", [
    "LoadFluxControlNet"
])

logging.info("MultiGPU: Starting Florence2 registration")
register_module("ComfyUI-Florence2", [
    "Florence2ModelLoader",
    "DownloadAndLoadFlorence2Model"
])

logging.info("MultiGPU: Starting LTXVideo registration")
register_module("ComfyUI-LTXVideo", [
    "LTXVLoader"
])

logging.info("MultiGPU: Starting MMAudio registration")
register_module("ComfyUI-MMAudio", [
    "MMAudioFeatureUtilsLoader",
    "MMAudioModelLoader",
    "MMAudioSampler"
])

logging.info("MultiGPU: Starting NF4 registration")
register_module("ComfyUI_bnb_nf4_fp4_Loaders", [
    "CheckpointLoaderNF4",
    "UNETLoaderNF4"
])


logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")