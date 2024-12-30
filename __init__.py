import time
import copy
import torch
import comfy.model_management
import os
import importlib.util
import logging

##############################################################################
# INITIAL SETUP
##############################################################################
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

##############################################################################
# OVERRIDE CLASS
##############################################################################
def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            # In case some node forgot "required"
            if "required" not in inputs:
                inputs["required"] = {}
            devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            inputs["required"]["device"] = (devices,)
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device="cpu", **kwargs):
            global current_device
            current_device = device
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)
    return NodeOverride

##############################################################################
# OUR LOCAL MAPPING OF MULTIGPU OVERRIDE NODES
# (No strict reason it has to be before the function defs, but it's typical.)
##############################################################################
NODE_CLASS_MAPPINGS = {}

##############################################################################
# PART 1: CORE NODES
##############################################################################
def register_core_nodes(core_node_names):
    """
    Uses ComfyUI's GLOBAL_NODE_CLASS_MAPPINGS to wrap core nodes.
    """
    logging.info("MultiGPU: Starting core node registration")
    try:
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
    except ImportError as e:
        logging.error(f"MultiGPU: Could not import ComfyUI global node mappings: {e}")
        return

    for node in core_node_names:
        if node in GLOBAL_NODE_CLASS_MAPPINGS:
            NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[node])
            logging.info(f"MultiGPU: Registered core node {node}")
        else:
            logging.info(f"MultiGPU: Core node '{node}' not found in ComfyUI global mappings")

##############################################################################
# PART 2: CUSTOM NODES (LOCAL DICTIONARY APPROACH, with extra debug logs)
##############################################################################
def register_module(module_path: str, target_nodes: list, local_map_name="NODE_CLASS_MAPPINGS"):
    """
    1) Load custom_nodes/<module_path>/__init__.py via importlib
    2) Grab the dictionary named `local_map_name` (default: 'NODE_CLASS_MAPPINGS') from that module
    3) For each node in target_nodes, if found in that local dictionary, wrap it
    4) No fallback to ComfyUI's global mappings
    """
    base_dir = os.path.join("custom_nodes", module_path)
    init_file = os.path.join(base_dir, "__init__.py")

    logging.info(f"MultiGPU: Checking custom node module at: {init_file}")
    if not os.path.exists(init_file):
        logging.info(f"MultiGPU: Module {module_path} not found or missing __init__.py, skipping.")
        return

    try:
        logging.info(f"MultiGPU: Found {module_path}, loading local dictionary from {init_file}")
        spec = importlib.util.spec_from_file_location(module_path, init_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.info(f"MultiGPU: Executed {module_path} initialization")
    except Exception as e:
        logging.error(f"MultiGPU: Error loading {module_path}: {e}")
        return

    # Grab that module's local dictionary (e.g. NODE_CLASS_MAPPINGS)
    local_map = getattr(module, local_map_name, None)
    if not local_map:
        logging.info(f"MultiGPU: {module_path} has no '{local_map_name}' dictionary, skipping override.")
        return

    # DEBUG: Show everything this node map provides
    all_defined_nodes = list(local_map.keys())
    logging.info(f"MultiGPU: {module_path} local dict keys: {all_defined_nodes}")

    # Wrap each node we want
    for node in target_nodes:
        if node in local_map:
            mgpu_class = override_class(local_map[node])
            NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = mgpu_class
            logging.info(f"MultiGPU: Successfully wrapped {node} from {module_path}")
        else:
            logging.info(f"MultiGPU: Node '{node}' not found in {module_path}'s local dictionary")


##############################################################################
# EXAMPLE USAGE
##############################################################################

# 1) CORE NODES
register_core_nodes([
    "UNETLoader",
    "VAELoader",
    "CLIPLoader",
    "DualCLIPLoader",
    "TripleCLIPLoader",
    "CheckpointLoaderSimple",
    "ControlNetLoader",
])

# 2) CUSTOM NODES
register_module("ComfyUI-GGUF", [
    "UnetLoaderGGUF",
    "UnetLoaderGGUFAdvanced",
    "CLIPLoaderGGUF",
    "DualCLIPLoaderGGUF",
    "TripleCLIPLoaderGGUF",
])

register_module("x-flux-comfyui", ["LoadFluxControlNet"])
register_module("ComfyUI-Florence2", ["Florence2ModelLoader", "DownloadAndLoadFlorence2Model"])
register_module("ComfyUI-LTXVideo", ["LTXVLoader"])
register_module("ComfyUI-MMAudio", ["MMAudioFeatureUtilsLoader", "MMAudioModelLoader", "MMAudioSampler"])
register_module("ComfyUI_bitsandbytes_NF4", ["CheckpointLoaderNF4"])

logging.info("MultiGPU: Registration complete.")
logging.info("MultiGPU: Final mappings: " + ", ".join(sorted(NODE_CLASS_MAPPINGS.keys())))
