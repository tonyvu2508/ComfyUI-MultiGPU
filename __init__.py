import time
import copy
import torch
import comfy.model_management
import os
from pathlib import Path  # Add this import
import importlib.util
import logging
import folder_paths

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

NODE_CLASS_MAPPINGS = {}


def check_module_exists(module_path):
    """Utility function to check if module exists"""
    full_path = os.path.join("custom_nodes", module_path, "__init__.py")
    logging.info(f"MultiGPU: Checking for module at {full_path}")
    
    if not os.path.exists(full_path):
        logging.info(f"MultiGPU: Module {module_path} not found - skipping")
        return False
        
    logging.info(f"MultiGPU: Found {module_path}, attempting to load")
    return True

def register_module(module_path, target_nodes):
    try:
        # For core nodes, skip module loading and just register from the global mappings
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

        # For custom nodes, try to load the module first
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

        # Use the module's local dictionary instead of the global one
        local_map_name = "NODE_CLASS_MAPPINGS"
        local_map = getattr(module, local_map_name, None)
        if not local_map:
            logging.info(f"MultiGPU: {module_path} has no '{local_map_name}' dictionary, skipping override.")
            return

        all_defined_nodes = list(local_map.keys())
        logging.info(f"MultiGPU: {module_path} local dict keys: {all_defined_nodes}")

        for node in target_nodes:
            if node in local_map:
                mgpu_class = override_class(local_map[node])
                NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = mgpu_class
                logging.info(f"MultiGPU: Successfully wrapped {node} from {module_path}")
            else:
                logging.info(f"MultiGPU: Node '{node}' not found in {module_path}'s local dictionary")

    except Exception as e:
        logging.info(f"MultiGPU: Error processing {module_path}: {str(e)}")

def register_LTXmodule(module_path, node_list):
    """Hard-coded registration for LTX Video nodes"""
    global NODE_CLASS_MAPPINGS
    
    if not check_module_exists(module_path):
        return
        
    class LTXVLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                                {"tooltip": "The name of the checkpoint (model) to load."}),
                    "dtype": (["bfloat16", "float32"], {"default": "bfloat16"})
                }
            }
        
        RETURN_TYPES = ("MODEL", "VAE")
        RETURN_NAMES = ("model", "vae")
        FUNCTION = "load"
        CATEGORY = "lightricks/LTXV"
        TITLE = "LTXV Loader"
        OUTPUT_NODE = False
        
        def load(self, ckpt_name, dtype):
            # Get original node instance
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
            
            # Use original node to load model and VAE
            model, vae = original_loader.load(ckpt_name, dtype)
            
            # Return original objects
            return (model, vae)

    ltx_nodes = {
        "LTXVLoader": LTXVLoader
    }

    for node_name in node_list:
        if node_name in ltx_nodes:
            NODE_CLASS_MAPPINGS[f"{node_name}MultiGPU"] = override_class(ltx_nodes[node_name])
            logging.info(f"MultiGPU: Registered hard-coded LTX node {node_name}")

def register_Florence2module(module_path, node_list):
    """Hard-coded registration for Florence2 nodes"""
    global NODE_CLASS_MAPPINGS
    
    if not check_module_exists(module_path):
        return
        
    class DownloadAndLoadFlorence2Model:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {
                "model": ([
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA',
                    'thwri/CogFlorence-2.1-Large',
                    'thwri/CogFlorence-2.2-Large',
                    'gokaygokay/Florence-2-SD3-Captioner',
                    'gokaygokay/Florence-2-Flux-Large',
                    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                    'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
                ], {"default": 'microsoft/Florence-2-base'}),
                "precision": (['fp16','bf16','fp32'], {"default": 'fp16'}),
                "attention": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'sdpa'}),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }}
        
        RETURN_TYPES = ("FL2MODEL",)
        RETURN_NAMES = ("florence2_model",)
        FUNCTION = "loadmodel"
        CATEGORY = "Florence2"
        
        def loadmodel(self, model, precision, attention, lora=None):
            # Get original node instance
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
            
            # Use original node to load model
            return original_loader.loadmodel(model, precision, attention, lora)

    class Florence2ModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {
                "model": ([item.name for item in Path(folder_paths.models_dir, "LLM").iterdir() if item.is_dir()], 
                         {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
                "precision": (['fp16','bf16','fp32'],),
                "attention": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'sdpa'}),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }}
        
        RETURN_TYPES = ("FL2MODEL",)
        RETURN_NAMES = ("florence2_model",)
        FUNCTION = "loadmodel"
        CATEGORY = "Florence2"
        
        def loadmodel(self, model, precision, attention, lora=None):
            # Get original node instance
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["Florence2ModelLoader"]()
            
            # Use original node to load model
            return original_loader.loadmodel(model, precision, attention, lora)

    florence2_nodes = {
        "Florence2ModelLoader": Florence2ModelLoader,
        "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model
    }

    for node_name in node_list:
        if node_name in florence2_nodes:
            NODE_CLASS_MAPPINGS[f"{node_name}MultiGPU"] = override_class(florence2_nodes[node_name])
            logging.info(f"MultiGPU: Registered hard-coded Florence2 node {node_name}")

# Register desired nodes
register_module("",                         ["UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader"])
register_module("ComfyUI-GGUF",             ["UnetLoaderGGUF","UnetLoaderGGUFAdvanced","CLIPLoaderGGUF","DualCLIPLoaderGGUF","TripleCLIPLoaderGGUF"])
register_module("x-flux-comfyui",           ["LoadFluxControlNet"])
register_Florence2module("ComfyUI-Florence2", ["Florence2ModelLoader", "DownloadAndLoadFlorence2Model"])
register_LTXmodule("ComfyUI-LTXVideo", ["LTXVLoader"])
register_module("ComfyUI-MMAudio",          ["MMAudioFeatureUtilsLoader","MMAudioModelLoader","MMAudioSampler"])
register_module("ComfyUI_bitsandbytes_NF4", ["CheckpointLoaderNF4",])

logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
