import os
import ast
import copy
import torch
import logging
from typing import Dict, Type

logging.info("MultiGPU: Initialization started")

def find_node_definition(module_path: str, node_name: str) -> Dict:
    """
    Finds a specific node class definition by searching Python files in the given path.
    """
    search_dir = os.path.join("custom_nodes", module_path)
    if not os.path.exists(search_dir):
        logging.info(f"MultiGPU: No custom_nodes directory {module_path}, skipping")
        return None
        
    py_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
                
    if not py_files:
        return None
            
    try:
        for file_path in py_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == node_name:
                    class_info = {
                        'input_types': None,
                        'return_types': None,
                        'function': None,
                        'category': None
                    }

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'INPUT_TYPES':
                            if any(d.id == 'classmethod' for d in item.decorator_list if isinstance(d, ast.Name)):
                                for stmt in item.body:
                                    if isinstance(stmt, ast.Return):
                                        try:
                                            class_info['input_types'] = ast.literal_eval(stmt.value)
                                        except:
                                            pass

                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    try:
                                        if target.id == 'RETURN_TYPES':
                                            class_info['return_types'] = ast.literal_eval(item.value)
                                        elif target.id == 'FUNCTION':
                                            class_info['function'] = ast.literal_eval(item.value)
                                        elif target.id == 'CATEGORY':
                                            class_info['category'] = ast.literal_eval(item.value)
                                    except:
                                        pass
                    
                    return class_info
                    
    except Exception as e:
        logging.error(f"MultiGPU: Error scanning for {node_name}: {str(e)}")
    
    return None

def create_multigpu_node(node_name: str, class_info: Dict) -> Type:
    """Creates a MultiGPU version of the node"""
    class MultiGPUNode:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = copy.deepcopy(class_info['input_types'])
            if inputs is None:
                inputs = {"required": {}, "optional": {}}
            elif "required" not in inputs:
                inputs["required"] = {}
            
            devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            inputs["required"]["device"] = (devices,)
            return inputs
            
        RETURN_TYPES = class_info['return_types'] if class_info['return_types'] is not None else tuple()
        FUNCTION = "override"
        CATEGORY = "multigpu"
        
        def override(self, *args, device="cpu", **kwargs):
            global current_device
            current_device = device
            return args if isinstance(args, tuple) else (args,)
            
    MultiGPUNode.__name__ = f"{node_name}MultiGPU"
    return MultiGPUNode

def override_class(cls):
    """Creates a MultiGPU version of a node class that preserves original functionality."""
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

def register_core_nodes(target_nodes: list):
    """Register MultiGPU versions of core ComfyUI nodes"""
    try:
        from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
        logging.info("MultiGPU: Processing core nodes")
        for node in target_nodes:
            if node in GLOBAL_NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS[node])
                logging.info(f"MultiGPU: Registered {node}MultiGPU")
            else:
                logging.info(f"MultiGPU: Core node {node} not found")
    except Exception as e:
        logging.error(f"MultiGPU: Error processing core nodes: {str(e)}")

def register_module(module_path: str, target_nodes: list):
    """Register MultiGPU versions of custom nodes"""
    try:
        # Handle custom nodes
        logging.info(f"MultiGPU: Processing module {module_path}")
        search_dir = os.path.join("custom_nodes", module_path)
        if not os.path.exists(search_dir):
            logging.info(f"MultiGPU: Module directory {module_path} not found, skipping")
            return

        # List all Python files in the module once
        py_files = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.basename(file))
        
        if py_files:
            logging.info(f"MultiGPU: Searching in {module_path}: {', '.join(sorted(py_files))}")
            
            for node in target_nodes:
                class_info = find_node_definition(module_path, node)
                if class_info:
                    NODE_CLASS_MAPPINGS[f"{node}MultiGPU"] = create_multigpu_node(node, class_info)
                    logging.info(f"MultiGPU: Registered {node}MultiGPU")
                
    except Exception as e:
        logging.error(f"MultiGPU: Error in {module_path}: {str(e)}")

# Initialize
NODE_CLASS_MAPPINGS = {}
current_device = None

# Register all modules
register_module("", ["UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader","TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader"])
register_module("ComfyUI-GGUF", ["UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "CLIPLoaderGGUF","DualCLIPLoaderGGUF", "TripleCLIPLoaderGGUF"])
register_module("x-flux-comfyui", ["LoadFluxControlNet"])
register_module("ComfyUI-Florence2", ["Florence2ModelLoader", "DownloadAndLoadFlorence2Model"])
register_module("ComfyUI-LTXVideo", ["LTXVLoader"])
register_module("ComfyUI-MMAudio", [    "MMAudioFeatureUtilsLoader", "MMAudioModelLoader", "MMAudioSampler"])
register_module("ComfyUI_bitsandbytes_NF4", ["CheckpointLoaderNF4"])

logging.info("MultiGPU: Registration complete")
logging.info(f"MultiGPU: Registered nodes: {', '.join(sorted(NODE_CLASS_MAPPINGS.keys()))}")