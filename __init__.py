import time
import copy
import torch
import sys
import comfy.model_management
import os
from pathlib import Path  # Add this import
import importlib.util
import logging
import folder_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("MultiGPU: Initialization started")

# Initialize the current device states and log them
current_device = comfy.model_management.get_torch_device()
current_offload_device = comfy.model_management.get_torch_device()
distorch_allocations = {}

logging.info(f"MultiGPU: Initial device set to {current_device}")
logging.info(f"MultiGPU: Initial offload device set to {current_offload_device}")

# Define and patch the device logic
def get_torch_device_patched():
    device = None
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
        or "cpu" in str(current_device).lower()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device(current_device)

    logging.info(f"MultiGPU: get_torch_device_patched invoked, returning {device}")
    return device

comfy.model_management.get_torch_device = get_torch_device_patched
logging.info(f"MultiGPU: Patched get_torch_device now returns {get_torch_device_patched()}")

def unet_offload_device_patched():
    device = None
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
        or "cpu" in str(current_offload_device).lower()
    ):
        device = torch.device("cpu")
    else:
        device = torch.device(current_offload_device)

    logging.info(f"MultiGPU: unet_offload_device_patched invoked, returning {device}")
    return device

comfy.model_management.unet_offload_device = unet_offload_device_patched
logging.info(f"MultiGPU: Patched unet_offload_device now returns {unet_offload_device_patched()}")

# Save the original patched logic for later restoration and log them
original_get_torch_device = comfy.model_management.get_torch_device
original_unet_offload_device = comfy.model_management.unet_offload_device
logging.info(f"MultiGPU: Saved original get_torch_device: {original_get_torch_device()}")
logging.info(f"MultiGPU: Saved original unet_offload_device: {original_unet_offload_device()}")

logging.info("MultiGPU: Device management logic initialized")


def analyze_ggml_loading(model):
    """
    Analyzes GGML model loading and determines device assignments.
    Returns device assignments with accurate memory calculations.
    """
    from collections import defaultdict

    # For testing - this would come from a global config in production
    DEVICE_RATIOS = {
        "cuda:0": 1,  # 1/9 of layers
        "cuda:1": 8   # 8/9 of layers
    }

    logging.info(f"MultiGPU: Input distorch_allocations: {distorch_allocations}")

    DEVICE_RATIOS_DISTORCH = {}

    # We will store extra info here so we can log a full table of devices,
    # including fraction%, total memory (GB), and allocated memory (GB).
    device_table = {}

    # 1) Primary compute device:
    primary_dev = current_device
    primary_dev_name = (
        f"{primary_dev.type}:{primary_dev.index}" if primary_dev.type != "cpu" else "cpu"
    )
    primary_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(primary_dev_name))
    primary_fraction = distorch_allocations.get("compute_device_alloc", 0.0)
    primary_alloc_bytes = primary_total_mem_bytes * primary_fraction

    DEVICE_RATIOS_DISTORCH[primary_dev_name] = primary_alloc_bytes
    device_table[primary_dev_name] = {
        "fraction": primary_fraction,
        "total_bytes": primary_total_mem_bytes,
        "alloc_bytes": primary_alloc_bytes,
    }

    # 2) Optional DistOrch devices (distorch1, distorch2, ...):
    i = 1
    while f"distorch{i}_device" in distorch_allocations:
        dev_key = f"distorch{i}_device"
        alloc_key = f"distorch{i}_alloc"

        dev_name = distorch_allocations[dev_key]
        dev_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(dev_name))
        dev_fraction = distorch_allocations.get(alloc_key, 0.0)
        dev_alloc_bytes = dev_total_mem_bytes * dev_fraction

        DEVICE_RATIOS_DISTORCH[dev_name] = dev_alloc_bytes
        device_table[dev_name] = {
            "fraction": dev_fraction,
            "total_bytes": dev_total_mem_bytes,
            "alloc_bytes": dev_alloc_bytes,
        }
        i += 1

    # 3) Always include CPU:
    cpu_dev_name = distorch_allocations.get("distorch_cpu", "cpu")
    cpu_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(cpu_dev_name))
    cpu_fraction = distorch_allocations.get("distorch_cpu_alloc", 0.0)
    cpu_alloc_bytes = cpu_total_mem_bytes * cpu_fraction

    DEVICE_RATIOS_DISTORCH[cpu_dev_name] = cpu_alloc_bytes
    device_table[cpu_dev_name] = {
        "fraction": cpu_fraction,
        "total_bytes": cpu_total_mem_bytes,
        "alloc_bytes": cpu_alloc_bytes,
    }

    # 4) Log the final constructed DEVICE_RATIOS_DISTORCH in raw bytes:
    logging.info(f"MultiGPU: Raw DEVICE_RATIOS_DISTORCH in bytes: {DEVICE_RATIOS_DISTORCH}")

    # 5) Print a table with device, fraction%, total mem (GB), allocated (GB)
    logging.info("\nDisTorch Device Analysis:")
    logging.info("========================================")
    fmt_str = "{:<10} {:>10} {:>12} {:>12}"
    logging.info(fmt_str.format("Device", "Alloc %", "Total (GB)", "Alloc (GB)"))
    logging.info("-" * 46)

    for dev in sorted(device_table.keys()):
        frac = device_table[dev]["fraction"]
        total_gb = device_table[dev]["total_bytes"] / (1024**3)
        alloc_gb = device_table[dev]["alloc_bytes"] / (1024**3)

        logging.info(fmt_str.format(
            dev,
            f"{frac * 100:.2f}%",    # fraction as a percentage
            f"{total_gb:.2f}",      # total memory in GB
            f"{alloc_gb:.2f}"       # allocated memory in GB
        ))

    logging.info("========================================\n")



    # Step 1: Memory Analysis
    device_properties = {}
    for device in DEVICE_RATIOS.keys():
        if device.startswith("cuda"):
            device_props = torch.cuda.get_device_properties(torch.device(device))
            device_properties[device] = {
                "total_memory": device_props.total_memory,
                "name": device_props.name
            }
            logging.info(f"ComfyUI-GGUF: Device {device} Memory: {device_props.total_memory / (1024 ** 3):.2f} GB")

    # Step 2: Layer Analysis
    layer_summary = {}
    layer_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    # First pass: collect layers and calculate total memory
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            layer_type = type(module).__name__
            layer_summary[layer_type] = layer_summary.get(layer_type, 0) + 1
            layer_list.append((name, module, layer_type))

            # Calculate memory for this layer
            layer_memory = 0
            if module.weight is not None:
                layer_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                layer_memory += module.bias.numel() * module.bias.element_size()
            
            memory_by_type[layer_type] += layer_memory
            total_memory += layer_memory

    # Step 3: Print Analysis Results as Tables
    logging.info("\nGGML Layer Analysis")
    logging.info("==================")
    
    # Layer Distribution Table
    format_str = "{:<12} {:>8} {:>12} {:>8}"
    logging.info("\nLayer Distribution:")
    logging.info(format_str.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logging.info("-" * 42)
    
    for layer_type, count in layer_summary.items():
        mem = memory_by_type[layer_type] / (1024 * 1024)  # MB
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(format_str.format(
            layer_type,
            str(count),
            f"{mem:.2f}",
            f"{mem_percent:.1f}%"
        ))

    # Step 4: Calculate Device Assignments
    total_ratio = sum(DEVICE_RATIOS.values())
    device_assignments = {device: [] for device in DEVICE_RATIOS.keys()}
    
    # Calculate layer counts for each device
    total_layers = len(layer_list)
    current_layer = 0
    
    for device, ratio in DEVICE_RATIOS.items():
        if device == list(DEVICE_RATIOS.keys())[-1]:
            # Last device gets all remaining layers
            device_layer_count = total_layers - current_layer
        else:
            device_layer_count = int((ratio / total_ratio) * total_layers)
        
        start_idx = current_layer
        end_idx = current_layer + device_layer_count
        device_assignments[device] = layer_list[start_idx:end_idx]
        current_layer += device_layer_count

    # Device Assignment Table with corrected memory calculations
    format_str = "{:<10} {:>8} {:>16} {:>10}"
    logging.info("\nDevice Assignments:")
    logging.info(format_str.format("Device", "Layers", "Memory (MB)", "% Total"))
    logging.info("-" * 46)
    
    total_assigned_memory = 0
    device_memories = {}
    
    # Calculate memory per device
    for device, layers in device_assignments.items():
        device_memory = 0
        # Calculate memory per layer type for this device
        for layer_type in layer_summary:
            type_layers = sum(1 for _, _, lt in layers if lt == layer_type)
            if layer_summary[layer_type] > 0:  # Avoid div by zero
                mem_per_layer = memory_by_type[layer_type] / layer_summary[layer_type]
                device_memory += mem_per_layer * type_layers
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    # Print device assignments with memory percentages
    for device, layers in device_assignments.items():
        mem_mb = device_memories[device] / (1024 * 1024)  # Convert to MB
        mem_percent = (device_memories[device] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(format_str.format(
            device,
            str(len(layers)),
            f"{mem_mb:.2f}",
            f"{mem_percent:.1f}%"
        ))

    # Verification log
    total_mb = total_memory / (1024 * 1024)
    assigned_mb = total_assigned_memory / (1024 * 1024)
    logging.info(f"\nMemory Verification:")
    logging.info(f"Total Model Memory: {total_mb:.2f} MB")
    logging.info(f"Total Assigned Memory: {assigned_mb:.2f} MB")
    if abs(total_mb - assigned_mb) > 0.01:  # Allow for minor floating point differences
        logging.warning(f"Memory assignment mismatch: {abs(total_mb - assigned_mb):.2f} MB difference")

    return {
        "device_assignments": device_assignments
    }

def get_device_list():
    import torch
    return ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]

class DeviceSelectorMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        return {
            "required": {
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0]})
            }
        }

    RETURN_TYPES = (get_device_list(),)
    RETURN_NAMES = ("device",)
    FUNCTION = "select_device"
    CATEGORY = "multigpu"

    def select_device(self, device):
        return (device,)

def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            global current_device
            if device is not None:
                current_device = device
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)

    return NodeOverride

def override_class_with_offload(cls):
    class NodeOverrideDiffSynth(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["offload_device"] = (devices, {"default": "cpu"})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, offload_device=None, **kwargs):
            global current_device
            global current_offload_device
            if device is not None:
                current_device = device
            if offload_device is not None:
                current_offload_device = offload_device
            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)

    return NodeOverrideDiffSynth

def override_class_with_distorch(cls):
    class NodeOverrideDisTorch(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = [d for d in get_device_list() if d != "cpu"]
            inputs["optional"] = inputs.get("optional", {})

            inputs["required"]["compute_device"] = (devices, {"default": devices[0], "tooltip": "Device model will use for computation"})
            inputs["required"]["compute_device_alloc"] = ("FLOAT", {"default": 0.15, "step": 0.01, "tooltip": "Fraction of memory NOT allocated to active latent space computation, recommended <= 15%"})

            for i in range(len(devices) - 1):
                inputs["optional"][f"distorch{i+1}_device"] = (devices, {"default": devices[i+1], "tooltip": f"Device for distorch{i+1} model layer VRAM allocation"})
                inputs["optional"][f"distorch{i+1}_alloc"] = ("FLOAT", {"default": 0.9, "step": 0.01, "tooltip": f"Fraction of memory allocated to distorch{i+1} model layer, recommended >= 90%"})

            inputs["optional"]["distorch_cpu"] = (["cpu"], {"default": "cpu", "tooltip": "Device for distorch CPU memory allocation"})
            inputs["optional"]["distorch_cpu_alloc"] = ("FLOAT", {"default": 0.0, "step": 0.01, "tooltip": "Fraction of memory allocated to distorch CPU memory (potentially slower than cuda)"})

            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, compute_device=None, **kwargs):
            global current_device
            global distorch_allocations

            current_device = compute_device
            distorch_allocations = {}

            for key, value in list(kwargs.items()):
                if key not in {"unet_name"}:
                    distorch_allocations[key] = kwargs.pop(key)

            logging.info(f"MultiGPU: DisTorch - distorch_allocations: {distorch_allocations}")

            fn = getattr(super(), cls.FUNCTION)
            return fn(*args, **kwargs)

    return NodeOverrideDisTorch

NODE_CLASS_MAPPINGS = {
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU
}

def check_module_exists(module_path):
    full_path = os.path.join("custom_nodes", module_path)
    logging.info(f"MultiGPU: Checking for module at {full_path}")
    
    if not os.path.exists(full_path):
        logging.info(f"MultiGPU: Module {module_path} not found - skipping")
        return False
        
    logging.info(f"MultiGPU: Found {module_path}, creating compatible MultiGPU nodes")
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

    except Exception as e:
        logging.info(f"MultiGPU: Error processing {module_path}: {str(e)}")

def register_LTXVLoaderMultiGPU():
    global NODE_CLASS_MAPPINGS
        
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
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
            return original_loader.load(ckpt_name, dtype)
        def _load_unet(self, load_device, offload_device, weights, num_latent_channels, dtype, config=None ):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
            return original_loader._load_unet(load_device, offload_device, weights, num_latent_channels, dtype, config=None )
        def _load_vae(self, weights, config=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
            return original_loader._load_vae(weights, config=None)
    NODE_CLASS_MAPPINGS["LTXVLoaderMultiGPU"] = override_class(LTXVLoader)

    logging.info(f"MultiGPU: Registered LTXVLoaderMultiGPU")

def register_Florence2ModelLoaderMultiGPU():
    global NODE_CLASS_MAPPINGS
    
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
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["Florence2ModelLoader"]()
            return original_loader.loadmodel(model, precision, attention, lora)
    NODE_CLASS_MAPPINGS["Florence2ModelLoaderMultiGPU"] = override_class(Florence2ModelLoader)
    logging.info(f"MultiGPU: Registered Florence2ModelLoaderMultiGPU")    
    
def register_DownloadAndLoadFlorence2ModelMultiGPU(): 
    global NODE_CLASS_MAPPINGS

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
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
            return original_loader.loadmodel(model, precision, attention, lora)
    NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2ModelMultiGPU"] = override_class(DownloadAndLoadFlorence2Model)
    logging.info(f"MultiGPU: Registered DownloadAndLoadFlorence2ModelMultiGPU")

def register_CheckpointLoaderNF4():
    global NODE_CLASS_MAPPINGS

    class CheckpointLoaderNF4:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                                }}
        RETURN_TYPES = ("MODEL", "CLIP", "VAE")
        FUNCTION = "load_checkpoint"

        CATEGORY = "loaders"


        def load_checkpoint(self, ckpt_name):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["CheckpointLoaderNF4"]()
            return original_loader.load_checkpoint(ckpt_name)
    
    NODE_CLASS_MAPPINGS["CheckpointLoaderNF4MultiGPU"] = override_class(CheckpointLoaderNF4)
    logging.info(f"MultiGPU: Registered CheckpointLoaderNF4MultiGPU")

def register_CheckpointLoaderNF4():
    global NODE_CLASS_MAPPINGS

    class CheckpointLoaderNF4:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                                }}
        RETURN_TYPES = ("MODEL", "CLIP", "VAE")
        FUNCTION = "load_checkpoint"

        CATEGORY = "loaders"


        def load_checkpoint(self, ckpt_name):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["CheckpointLoaderNF4"]()
            return original_loader.load_checkpoint(ckpt_name)
    
    NODE_CLASS_MAPPINGS["CheckpointLoaderNF4MultiGPU"] = override_class(CheckpointLoaderNF4)
    logging.info(f"MultiGPU: Registered CheckpointLoaderNF4MultiGPU")

def register_LoadFluxControlNetMultiGPU():
    global NODE_CLASS_MAPPINGS

    class LoadFluxControlNet:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"model_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                                "controlnet_path": (folder_paths.get_filename_list("xlabs_controlnets"), ),
                                }}

        RETURN_TYPES = ("FluxControlNet",)
        RETURN_NAMES = ("ControlNet",)
        FUNCTION = "loadmodel"
        CATEGORY = "XLabsNodes"

        def loadmodel(self, model_name, controlnet_path):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["LoadFluxControlNet"]()
            return original_loader.loadmodel(model_name, controlnet_path)
        
    NODE_CLASS_MAPPINGS["LoadFluxControlNetMultiGPU"] = override_class(LoadFluxControlNet)
    logging.info(f"MultiGPU: Registered LoadFluxControlNetMultiGPU")

def register_MMAudioModelLoaderMultiGPU():

    global NODE_CLASS_MAPPINGS

    class MMAudioModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' -folder",}),
                
                "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                },
            }

        RETURN_TYPES = ("MMAUDIO_MODEL",)
        RETURN_NAMES = ("mmaudio_model", )
        FUNCTION = "loadmodel"
        CATEGORY = "MMAudio"

        def loadmodel(self, mmaudio_model, base_precision):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["MMAudioModelLoader"]()
            return original_loader.loadmodel(mmaudio_model, base_precision)
        
    NODE_CLASS_MAPPINGS["MMAudioModelLoaderMultiGPU"] = override_class(MMAudioModelLoader)
    logging.info(f"MultiGPU: Registered MMAudioModelLoaderMultiGPU")

def register_MMAudioModelLoaderMultiGPU():

    global NODE_CLASS_MAPPINGS

    class MMAudioModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' -folder",}),
                
                "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                },
            }

        RETURN_TYPES = ("MMAUDIO_MODEL",)
        RETURN_NAMES = ("mmaudio_model", )
        FUNCTION = "loadmodel"
        CATEGORY = "MMAudio"

        def loadmodel(self, mmaudio_model, base_precision):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["MMAudioModelLoader"]()
            return original_loader.loadmodel(mmaudio_model, base_precision)
        
    NODE_CLASS_MAPPINGS["MMAudioModelLoaderMultiGPU"] = override_class(MMAudioModelLoader)
    logging.info(f"MultiGPU: Registered MMAudioModelLoaderMultiGPU")

def register_MMAudioFeatureUtilsLoaderMultiGPU():

    global NODE_CLASS_MAPPINGS


    class MMAudioFeatureUtilsLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "vae_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                    "synchformer_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                    "clip_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                },
                "optional": {
                "bigvgan_vocoder_model": ("VOCODER_MODEL", {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                    "mode": (["16k", "44k"], {"default": "44k"}),
                    "precision": (["fp16", "fp32", "bf16"],
                        {"default": "fp16"}
                    ),
                }
            }

        RETURN_TYPES = ("MMAUDIO_FEATUREUTILS",)
        RETURN_NAMES = ("mmaudio_featureutils", )
        FUNCTION = "loadmodel"
        CATEGORY = "MMAudio"

        def loadmodel(self, vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["MMAudioFeatureUtilsLoader"]()
            return original_loader.loadmodel(vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model)
    
    NODE_CLASS_MAPPINGS["MMAudioFeatureUtilsLoaderMultiGPU"] = override_class(MMAudioFeatureUtilsLoader)
    logging.info(f"MultiGPU: Registered MMAudioFeatureUtilsLoaderMultiGPU")

def register_MMAudioSamplerMultiGPU():

    global NODE_CLASS_MAPPINGS

    class MMAudioSampler:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "mmaudio_model": ("MMAUDIO_MODEL",),
                    "feature_utils": ("MMAUDIO_FEATUREUTILS",),
                    "duration": ("FLOAT", {"default": 8, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                    "steps": ("INT", {"default": 25, "step": 1, "tooltip": "Number of steps to interpolate"}),
                    "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompt": ("STRING", {"default": "", "multiline": True} ),
                    "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                    "mask_away_clip": ("BOOLEAN", {"default": False, "tooltip": "If true, the clip video will be masked away"}),
                    "force_offload": ("BOOLEAN", {"default": True, "tooltip": "If true, the model will be offloaded to the offload device"}),
                },
                "optional": {
                    "images": ("IMAGE",),
                },
            }

        RETURN_TYPES = ("AUDIO",)
        RETURN_NAMES = ("audio", )
        FUNCTION = "sample"
        CATEGORY = "MMAudio"

        def sample(self, mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["MMAudioSampler"]()
            return original_loader.sample(mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images)
        
    NODE_CLASS_MAPPINGS["MMAudioSamplerMultiGPU"] = override_class(MMAudioSampler)
    logging.info(f"MultiGPU: Registered MMAudioSamplerMultiGPU")

def register_UnetLoaderGGUFMultiGPU():
    global NODE_CLASS_MAPPINGS

    # First define the base UnetLoaderGGUF class
    class UnetLoaderGGUF:
        @classmethod
        def INPUT_TYPES(s):
            unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
            return {
                "required": {
                    "unet_name": (unet_names,),
                }
            }

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_unet"
        CATEGORY = "bootleg"
        TITLE = "Unet Loader (GGUF)"

        def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            return original_loader.load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)

    # Create the MultiGPU version of the base class
    UnetLoaderGGUFMultiGPU = override_class(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFMultiGPU"] = UnetLoaderGGUFMultiGPU
    logging.info(f"MultiGPU: Registered UnetLoaderGGUFMultiGPU")

    # Now create the advanced version that inherits from the MultiGPU base class
    class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
        @classmethod
        def INPUT_TYPES(s):
            unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
            return {
                "required": {
                    "unet_name": (unet_names,),
                    "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                    "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                    "patch_on_device": ("BOOLEAN", {"default": False}),
                }
            }
        TITLE = "Unet Loader (GGUF/Advanced)"

    # Create the MultiGPU version of the advanced class
    UnetLoaderGGUFAdvancedMultiGPU = override_class(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedMultiGPU"] = UnetLoaderGGUFAdvancedMultiGPU
    logging.info(f"MultiGPU: Registered UnetLoaderGGUFAdvancedMultiGPU")

def register_UnetLoaderGGUFDisTorchMultiGPU():
    global NODE_CLASS_MAPPINGS

    # First define the base UnetLoaderGGUFDisTorch class
    class UnetLoaderGGUFDisTorch:
        @classmethod
        def INPUT_TYPES(s):
            unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
            return {
                "required": {
                    "unet_name": (unet_names,),
                }
            }

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load_unet"
        CATEGORY = "bootleg"
        TITLE = "Unet Loader (GGUFDisTorch)"

        def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
            from nodes import NODE_CLASS_MAPPINGS
            logging.info("MultiGPU: GGUFDisTorch - Starting GGUFDisTorch UNet load")
            
            # Get the correct module through the original loader
            original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
            module = sys.modules[original_loader.__module__]
            logging.info(f"MultiGPU: GGUFDisTorch - Got GGUF module: {module}")
            
            if not hasattr(module.GGUFModelPatcher, '_patched'):
                original_load = module.GGUFModelPatcher.load
                logging.info("MultiGPU: GGUFDisTorch - GGUF ModelPatcher not yet patched, applying patch")
                
                def new_load(self, *args, force_patch_weights=False, **kwargs):
                    logging.info("MultiGPU: GGUFDisTorch - Entering patched GGUFDisTorch load function")

                    # Save the current device states and logic
                    global current_device, current_offload_device
                    original_current_device = current_device
                    original_current_offload_device = current_offload_device

                    try:
                        # Temporarily override the device logic for this load
                        current_device = torch.device("cuda:0")
                        current_offload_device = torch.device("cuda:1")

                        logging.info(f"MultiGPU: GGUFDisTorch - Overriding current_device to {current_device}")
                        logging.info(f"MultiGPU: GGUFDisTorch - Overriding current_offload_device to {current_offload_device}")

                        # Call the original load function with the temporary overrides
                        super(module.GGUFModelPatcher, self).load(*args, force_patch_weights=True, **kwargs)

                        if not self.mmap_released:
                            logging.info("MultiGPU: GGUFDisTorch - Processing mmap release")
                            linked = []
                            
                            # Debug the lowvram check
                            lowvram_value = kwargs.get("lowvram_model_memory", 0)
                            logging.info(f"MultiGPU: GGUFDisTorch - lowvram_model_memory value: {lowvram_value}")
                            
                            if lowvram_value > 0:
                                logging.info("MultiGPU: GGUFDisTorch - Entering module scanning")
                                module_count = 0
                                for n, m in self.model.named_modules():
                                    module_count += 1
                                    if hasattr(m, "weight"):
                                        device = getattr(m.weight, "device", None)
                                        # logging.info(f"MultiGPU: GGUFDisTorch - Module {n} on device {device}, offload_device is {self.offload_device}")
                                        if device == self.offload_device:
                                            linked.append((n, m))
                                logging.info(f"MultiGPU: GGUFDisTorch - Scanned {module_count} total modules")
                            else:
                                logging.info("MultiGPU: GGUFDisTorch - Skipped module scanning due to lowvram check")
                            
                            if linked:
                                logging.info(f"MultiGPU: GGUFDisTorch - Found {len(linked)} linked modules")
                                device_assignments = analyze_ggml_loading(self.model)['device_assignments']
                                for device, layers in device_assignments.items():
                                    target_device = torch.device(device)
                                    logging.info(f"MultiGPU: GGUFDisTorch - Moving {len(layers)} layers to {device}")
                                    for n, m, _ in layers:
                                        try:
                                            m.to(self.load_device).to(target_device)
                                         #   logging.info(f"MultiGPU: GGUFDisTorch - Successfully moved layer {n} to {device}")
                                        except Exception as e:
                                            logging.error(f"MultiGPU: GGUFDisTorch - Error moving layer {n} to {device}: {str(e)}")
                                self.mmap_released = True
                                logging.info("MultiGPU: GGUFDisTorch - mmap release complete")
                    
                    finally:
                        # Restore the original device states
                        current_device = original_current_device
                        current_offload_device = original_current_offload_device

                        logging.info(f"MultiGPU: GGUFDisTorch - Restored current_device to {current_device}")
                        logging.info(f"MultiGPU: GGUFDisTorch - Restored current_offload_device to {current_offload_device}")

                module.GGUFModelPatcher.load = new_load
                module.GGUFModelPatcher._patched = True
                logging.info("MultiGPU: GGUFDisTorch - Successfully patched GGUF ModelPatcher")
            else:
                logging.info("MultiGPU: GGUFDisTorch - GGUF ModelPatcher already patched")

            logging.info("MultiGPU: GGUFDisTorch - Calling original GGUF loader")
            loader_instance = original_loader()
            return loader_instance.load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)


    # Create the MultiGPU version of the base class
    UnetLoaderGGUFDisTorchMultiGPU = override_class_with_distorch(UnetLoaderGGUFDisTorch)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorchMultiGPU"] = UnetLoaderGGUFDisTorchMultiGPU
    logging.info(f"MultiGPU: Registered UnetLoaderGGUFDisTorchMultiGPU")

def register_CLIPLoaderGGUFMultiGPU():
    global NODE_CLASS_MAPPINGS

    class CLIPLoaderGGUF:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "clip_name": (s.get_filename_list(),),
                    "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv"],),
                }
            }

        RETURN_TYPES = ("CLIP",)
        FUNCTION = "load_clip"
        CATEGORY = "bootleg"
        TITLE = "CLIPLoader (GGUF)"

        @classmethod
        def get_filename_list(s):
            files = []
            files += folder_paths.get_filename_list("clip")
            files += folder_paths.get_filename_list("clip_gguf")
            return sorted(files)

        def load_data(self, ckpt_paths):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
            return original_loader.load_data(ckpt_paths)

        def load_patcher(self, clip_paths, clip_type, clip_data):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
            return original_loader.load_patcher(clip_paths, clip_type, clip_data)

        def load_clip(self, clip_name, type="stable_diffusion"):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
            return original_loader.load_clip(clip_name, type)

    # Create the MultiGPU version of the base class
    CLIPLoaderGGUFMultiGPU = override_class(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFMultiGPU"] = CLIPLoaderGGUFMultiGPU
    logging.info(f"MultiGPU: Registered CLIPLoaderGGUFMultiGPU")

    # Now create the advanced version that inherits from the MultiGPU base class

    class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
        @classmethod
        def INPUT_TYPES(s):
            file_options = (s.get_filename_list(), )
            return {
                "required": {
                    "clip_name1": file_options,
                    "clip_name2": file_options,
                    "type": (("sdxl", "sd3", "flux", "hunyuan_video"),),
                }
            }

        TITLE = "DualCLIPLoader (GGUF)"

        def load_clip(self, clip_name1, clip_name2, type):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
            return original_loader.load_clip(clip_name1, clip_name2, type)
    # Create the MultiGPU version of the advanced class
    DualCLIPLoaderGGUFMultiGPU = override_class(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFMultiGPU"] = DualCLIPLoaderGGUFMultiGPU
    logging.info(f"MultiGPU: Registered DualCLIPLoaderGGUFMultiGPU")

    class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
        @classmethod
        def INPUT_TYPES(s):
            file_options = (s.get_filename_list(), )
            return {
                "required": {
                    "clip_name1": file_options,
                    "clip_name2": file_options,
                    "clip_name3": file_options,
                }
            }

        TITLE = "TripleCLIPLoader (GGUF)"

        def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUF"]()
            return original_loader.load_clip(clip_name1, clip_name2, clip_name3, type)
    # Create the MultiGPU version of the advanced class
    TripleCLIPLoaderGGUFMultiGPU = override_class(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFMultiGPU"] = TripleCLIPLoaderGGUFMultiGPU
    logging.info(f"MultiGPU: Registered TripleCLIPLoaderGGUFMultiGPU")
 
def register_PulidModelLoader():

    global NODE_CLASS_MAPPINGS

    class PulidModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { "pulid_file": (folder_paths.get_filename_list("pulid"), )}}

        RETURN_TYPES = ("PULID",)
        FUNCTION = "load_model"
        CATEGORY = "pulid"

        def load_model(self, pulid_file):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["PulidModelLoader"]()
            return original_loader.load_model(pulid_file)
        
    NODE_CLASS_MAPPINGS["PulidModelLoaderMultiGPU"] = override_class(PulidModelLoader)
    logging.info(f"MultiGPU: Registered PulidModelLoaderMultiGPU")

def register_PulidInsightFaceLoader():

    global NODE_CLASS_MAPPINGS

    class PulidInsightFaceLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
                },
            }

        RETURN_TYPES = ("FACEANALYSIS",)
        FUNCTION = "load_insightface"
        CATEGORY = "pulid"

        def load_insightface(self, provider):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["PulidInsightFaceLoader"]()
            return original_loader.load_insightface(provider)
        
    NODE_CLASS_MAPPINGS["PulidInsightFaceLoaderMultiGPU"] = override_class(PulidInsightFaceLoader)
    logging.info(f"MultiGPU: Registered PulidInsightFaceLoaderMultiGPU")

def register_PulidEvaClipLoader():

    global NODE_CLASS_MAPPINGS

    class PulidEvaClipLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {},
            }

        RETURN_TYPES = ("EVA_CLIP",)
        FUNCTION = "load_eva_clip"
        CATEGORY = "pulid"

        def load_eva_clip(self):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]()
            return original_loader.load_eva_clip()
        
    NODE_CLASS_MAPPINGS["PulidEvaClipLoaderMultiGPU"] = override_class(PulidEvaClipLoader)
    logging.info(f"MultiGPU: Registered PulidEvaClipLoaderMultiGPU")

def register_HyVideoModelLoader():
    global NODE_CLASS_MAPPINGS

    # Keep original MultiGPU wrapper unchanged
    class HyVideoModelLoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                    "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
                    "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_scaled', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
                    "load_device": (["main_device"], {"default": "main_device"}),
                },
                "optional": {
                    "attention_mode": ([
                        "sdpa",
                        "flash_attn_varlen",
                        "sageattn_varlen",
                        "comfy",
                    ], {"default": "flash_attn"}),
                    "compile_args": ("COMPILEARGS", ),
                    "block_swap_args": ("BLOCKSWAPARGS", ),
                    "lora": ("HYVIDLORA", {"default": None}),
                    "auto_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable auto offloading for reduced VRAM usage, implementation from DiffSynth-Studio, slightly different from block swapping and uses even less VRAM, but can be slower as you can't define how much VRAM to use"}),
                }
            }

        RETURN_TYPES = ("HYVIDEOMODEL",)
        RETURN_NAMES = ("model", )
        FUNCTION = "loadmodel"
        CATEGORY = "HunyuanVideoWrapper"

        def loadmodel(self, model, base_precision, load_device, quantization, compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, auto_cpu_offload=False):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["HyVideoModelLoader"]()
            return original_loader.loadmodel(model, base_precision, load_device, quantization, compile_args, attention_mode, block_swap_args, lora, auto_cpu_offload)

    # Add new DiffSynth-style node
    class HyVideoModelLoaderDiffSynth:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                    "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
                    "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_scaled', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], 
                                   {"default": 'disabled', "tooltip": "optional quantization method"}),
                },
                "optional": {
                    "attention_mode": ([
                        "sdpa",
                        "flash_attn_varlen",
                        "sageattn_varlen",
                        "comfy",
                    ], {"default": "flash_attn"}),
                    "compile_args": ("COMPILEARGS", ),
                    "block_swap_args": ("BLOCKSWAPARGS", ),
                    "lora": ("HYVIDLORA", {"default": None}),
                }
            }

        RETURN_TYPES = ("HYVIDEOMODEL",)
        RETURN_NAMES = ("model", )
        FUNCTION = "loadmodel"
        CATEGORY = "HunyuanVideoWrapper"

        def loadmodel(self, model, base_precision, quantization, compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["HyVideoModelLoader"]()
            # Use DiffSynth's auto offloading approach
            return original_loader.loadmodel(model, base_precision, "main_device", quantization, 
                                          compile_args, attention_mode, block_swap_args, lora, 
                                          auto_cpu_offload=True)

    # Register both with MultiGPU wrapper
    NODE_CLASS_MAPPINGS["HyVideoModelLoaderMultiGPU"] = override_class(HyVideoModelLoader)
    NODE_CLASS_MAPPINGS["HyVideoModelLoaderDiffSynthMultiGPU"] = override_class_with_offload(HyVideoModelLoaderDiffSynth)
    
    logging.info(f"MultiGPU: Registered HyVideoModelLoader nodes")

def register_HyVideoVAELoader():

    global NODE_CLASS_MAPPINGS

    class HyVideoVAELoader:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
                },
                "optional": {
                    "precision": (["fp16", "fp32", "bf16"],
                        {"default": "bf16"}
                    ),
                    "compile_args":("COMPILEARGS", ),
                }
            }

        RETURN_TYPES = ("VAE",)
        RETURN_NAMES = ("vae", )
        FUNCTION = "loadmodel"
        CATEGORY = "HunyuanVideoWrapper"
        DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

        def loadmodel(self, model_name, precision, compile_args=None):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["HyVideoVAELoader"]()
            return original_loader.loadmodel(model_name, precision, compile_args)
        
    NODE_CLASS_MAPPINGS["HyVideoVAELoaderMultiGPU"] = override_class(HyVideoVAELoader)
    logging.info(f"MultiGPU: Registered HyVideoVAELoaderMultiGPU")

def register_DownloadAndLoadHyVideoTextEncoder():

    global NODE_CLASS_MAPPINGS

    class DownloadAndLoadHyVideoTextEncoder:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "llm_model": (["Kijai/llava-llama-3-8b-text-encoder-tokenizer","xtuner/llava-llama-3-8b-v1_1-transformers"],),
                    "clip_model": (["disabled","openai/clip-vit-large-patch14",],),
                    "precision": (["fp16", "fp32", "bf16"],
                        {"default": "bf16"}
                    ),
                },
                "optional": {
                    "apply_final_norm": ("BOOLEAN", {"default": False}),
                    "hidden_state_skip_layer": ("INT", {"default": 2}),
                    "quantization": (['disabled', 'bnb_nf4', "fp8_e4m3fn"], {"default": 'disabled'}),
                }
            }

        RETURN_TYPES = ("HYVIDTEXTENCODER",)
        RETURN_NAMES = ("hyvid_text_encoder", )
        FUNCTION = "loadmodel"
        CATEGORY = "HunyuanVideoWrapper"
        DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

        def loadmodel(self, llm_model, clip_model, precision,  apply_final_norm=False, hidden_state_skip_layer=2, quantization="disabled"):
            from nodes import NODE_CLASS_MAPPINGS
            original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadHyVideoTextEncoder"]()
            return original_loader.loadmodel(llm_model, clip_model, precision, apply_final_norm, hidden_state_skip_layer, quantization)
        
    NODE_CLASS_MAPPINGS["DownloadAndLoadHyVideoTextEncoderMultiGPU"] = override_class(DownloadAndLoadHyVideoTextEncoder)
    logging.info(f"MultiGPU: Registered DownloadAndLoadHyVideoTextEncoderMultiGPU")

# Register desired nodes
register_module("",                         ["UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader"])

if check_module_exists("ComfyUI-LTXVideo"):
    register_LTXVLoaderMultiGPU()
if check_module_exists("ComfyUI-Florence2"):
    register_Florence2ModelLoaderMultiGPU()
    register_DownloadAndLoadFlorence2ModelMultiGPU()
if check_module_exists("ComfyUI_bitsandbytes_NF4"):
    register_CheckpointLoaderNF4()
if check_module_exists("x-flux-comfyui"):
    register_LoadFluxControlNetMultiGPU()
if check_module_exists("ComfyUI-MMAudio"):
    register_MMAudioModelLoaderMultiGPU()
    register_MMAudioFeatureUtilsLoaderMultiGPU()
    register_MMAudioSamplerMultiGPU()
if check_module_exists("ComfyUI-GGUF"):
    register_UnetLoaderGGUFMultiGPU()
    register_UnetLoaderGGUFDisTorchMultiGPU()
    register_CLIPLoaderGGUFMultiGPU()
if check_module_exists("PuLID_ComfyUI"):
    register_PulidModelLoader()
    register_PulidInsightFaceLoader()
    register_PulidEvaClipLoader()
if check_module_exists("ComfyUI-HunyuanVideoWrapper"):
    register_HyVideoModelLoader()
    register_HyVideoVAELoader()
    register_DownloadAndLoadHyVideoTextEncoder()

logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
