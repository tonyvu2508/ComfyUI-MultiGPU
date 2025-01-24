import sys
import logging
import torch
from collections import defaultdict
import comfy.model_management
import copy

model_allocation_store = {}

def create_model_hash(model, caller):
   import hashlib
   
   logging.info(f"\nMultiGPU: Hash creation from {caller}")

   # Model type, size, and first 3 layers as identifier
   model_type = type(model.model).__name__
   model_size = model.model_size()
   first_layers = str(list(model.model_state_dict().keys())[:3])
   
   logging.info(f"MultiGPU: Type: {model_type}")
   logging.info(f"MultiGPU: Size: {model_size}")
   logging.info(f"MultiGPU: Layer sample: {first_layers}")

   identifier = f"{model_type}_{model_size}_{first_layers}"
   final_hash = hashlib.sha256(identifier.encode()).hexdigest()
   logging.info(f"MultiGPU: Hash: {final_hash}")
   
   return final_hash

# Add to both places as specified  

def register_patched_ggufmodelpatcher(node_instance):
    from nodes import NODE_CLASS_MAPPINGS
    original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
    module = sys.modules[original_loader.__module__]

    if not hasattr(module.GGUFModelPatcher, '_patched'):
        original_load = module.GGUFModelPatcher.load
        logging.info("MultiGPU: GGUFDisTorch - GGUF ModelPatcher not yet patched, applying patch")

        def new_load(self, *args, force_patch_weights=False, **kwargs):
            global model_allocation_store

            super(module.GGUFModelPatcher, self).load(*args, force_patch_weights=True, **kwargs)
            debug_hash = create_model_hash(self, "patcher") 
            linked = []
            module_count = 0
            for n, m in self.model.named_modules():
                module_count += 1
                if hasattr(m, "weight"):
                    device = getattr(m.weight, "device", None)
                    #logging.info(f"MultiGPU: GGUFDisTorch - Weight Module {n} on device {device}, offload_device is {self.offload_device}")
                    if device is not None:
                        linked.append((n, m))
                        continue
                if hasattr(m, "bias"):
                    device = getattr(m.bias, "device", None)
                    #logging.info(f"MultiGPU: GGUFDisTorch - Bias Module {n} on device {device}, offload_device is {self.offload_device}")
                    if device is not None:
                        linked.append((n, m))
                        continue
                #logging.info(f"MultiGPU: GGUFDisTorch - Found {len(linked)} linked modules out of {module_count} total modules")
            if linked:
                    #logging.info(f"MultiGPU: GGUFDisTorch - Found {len(linked)} linked modules, computing reallocation")
                    if hasattr(self, 'model'):
                        debug_hash = create_model_hash(self, "patcher")
                        debug_allocations = model_allocation_store.get(debug_hash)
                        logging.info(f"MultiGPU: Hash lookup - Found allocations: {debug_allocations}")
                        logging.info(f"MultiGPU: LOOKUP - Hash {debug_hash}")
                        if debug_allocations:
                            logging.info(f"MultiGPU: FOUND - Hash matches, using allocations: {debug_allocations}")
                        else:
                            logging.info(f"MultiGPU: MISS - Hash not found in store")
                    device_assignments = analyze_ggml_loading(self.model, node_instance.distorch_allocations)['device_assignments']
                    for device, layers in device_assignments.items():
                        #logging.info(f"MultiGPU: GGUFDisTorch - Moving {len(layers)} layers to {device}")
                        target_device = torch.device(device)
                        #logging.info(f"MultiGPU: GGUFDisTorch - Moving {len(layers)} layers to {device}")
                        for n, m, _ in layers:
                            m.to(self.load_device).to(target_device)

                    self.mmap_released = True
                    logging.info("MultiGPU: GGUFDisTorch - self.mmap_released = True")


        module.GGUFModelPatcher.load = new_load
        module.GGUFModelPatcher._patched = True
        logging.info("MultiGPU: GGUFDisTorch - Successfully patched GGUF ModelPatcher")
    else:
        logging.info("MultiGPU: GGUFDisTorch - GGUF ModelPatcher already patched")

def analyze_ggml_loading(model, distorch_allocations):

    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    primary_dev_name = distorch_allocations.get("compute_device")
    primary_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(primary_dev_name))
    primary_fraction = distorch_allocations.get("compute_device_alloc", 0.0)
    primary_alloc_gb = (primary_total_mem_bytes * primary_fraction) / (1024**3)
    DEVICE_RATIOS_DISTORCH[primary_dev_name] = primary_alloc_gb
    device_table[primary_dev_name] = {"fraction": primary_fraction,"total_gb": primary_total_mem_bytes / (1024**3),"alloc_gb": primary_alloc_gb}

    i = 1
    while f"distorch{i}_device" in distorch_allocations:
        dev_key = f"distorch{i}_device"
        alloc_key = f"distorch{i}_alloc"
        dev_name = distorch_allocations[dev_key]
        dev_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(dev_name))
        dev_fraction = distorch_allocations.get(alloc_key, 0.0)
        dev_alloc_gb = (dev_total_mem_bytes * dev_fraction) / (1024**3)
        DEVICE_RATIOS_DISTORCH[dev_name] = dev_alloc_gb
        device_table[dev_name] = {"fraction": dev_fraction,"total_gb": dev_total_mem_bytes / (1024**3),"alloc_gb": dev_alloc_gb}
        i += 1

    cpu_dev_name = distorch_allocations.get("distorch_cpu", "cpu")
    cpu_total_mem_bytes = comfy.model_management.get_total_memory(torch.device(cpu_dev_name))
    cpu_fraction = distorch_allocations.get("distorch_cpu_alloc", 0.0)
    cpu_alloc_gb = (cpu_total_mem_bytes * cpu_fraction) / (1024**3)
    DEVICE_RATIOS_DISTORCH[cpu_dev_name] = cpu_alloc_gb
    device_table[cpu_dev_name] = {"fraction": cpu_fraction,"total_gb": cpu_total_mem_bytes / (1024**3),"alloc_gb": cpu_alloc_gb}

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_alloc = "{:<12}{:>10}{:>14}{:>10}"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(eq_line)
    logging.info("               DisTorch Analysis")
    logging.info(eq_line)
    logging.info(dash_line)
    logging.info("          DisTorch Device Allocations")
    logging.info(dash_line)
    logging.info(fmt_alloc.format("Device", "Alloc %", "Total (GB)", " Alloc (GB)"))
    logging.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_devices:
        frac = device_table[dev]["fraction"]
        tot_gb = device_table[dev]["total_gb"]
        alloc_gb = device_table[dev]["alloc_gb"]
        logging.info(fmt_alloc.format(dev,f"{int(frac * 100)}%",f"{tot_gb:.2f}",f"{alloc_gb:.2f}"))

    logging.info(dash_line)

    layer_summary = {}
    layer_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            layer_type = type(module).__name__
            layer_summary[layer_type] = layer_summary.get(layer_type, 0) + 1
            layer_list.append((name, module, layer_type))
            layer_memory = 0
            if module.weight is not None:
                layer_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                layer_memory += module.bias.numel() * module.bias.element_size()
            memory_by_type[layer_type] += layer_memory
            total_memory += layer_memory

    logging.info("     DisTorch GGML Layer Distribution")
    logging.info(dash_line)
    fmt_layer = "{:<12}{:>10}{:>14}{:>10}"
    logging.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logging.info(dash_line)
    for layer_type, count in layer_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(fmt_layer.format(layer_type,str(count),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logging.info(dash_line)

    nonzero_devices = [d for d, r in DEVICE_RATIOS_DISTORCH.items() if r > 0]
    nonzero_total_ratio = sum(DEVICE_RATIOS_DISTORCH[d] for d in nonzero_devices)
    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH.keys()}
    total_layers = len(layer_list)
    current_layer = 0

    for idx, device in enumerate(nonzero_devices):
        ratio = DEVICE_RATIOS_DISTORCH[device]
        if idx == len(nonzero_devices) - 1:
            device_layer_count = total_layers - current_layer
        else:
            device_layer_count = int((ratio / nonzero_total_ratio) * total_layers)
        start_idx = current_layer
        end_idx = current_layer + device_layer_count
        device_assignments[device] = layer_list[start_idx:end_idx]
        current_layer += device_layer_count

    logging.info("    DisTorch Final Device/Layer Assignments")
    logging.info(dash_line)
    fmt_assign = "{:<12}{:>10}{:>14}{:>10}"
    logging.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logging.info(dash_line)
    total_assigned_memory = 0
    device_memories = {}
    for device, layers in device_assignments.items():
        device_memory = 0
        for layer_type in layer_summary:
            type_layers = sum(1 for _, _, lt in layers if lt == layer_type)
            if layer_summary[layer_type] > 0:
                mem_per_layer = memory_by_type[layer_type] / layer_summary[layer_type]
                device_memory += mem_per_layer * type_layers
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    sorted_assignments = sorted(device_assignments.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        layers = device_assignments[dev]
        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(fmt_assign.format(dev,str(len(layers)),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logging.info(dash_line)

    return {"device_assignments": device_assignments}


def override_class_with_distorch(cls):
    from . import register_patched_ggufmodelpatcher
    from . import get_device_list
    import copy
    import logging

    class NodeOverrideDisTorch(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.distorch_allocations = {}
            self.distorch_compute_device = None

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


        def override(self, *args, **kwargs):
            global current_device, model_allocation_store
            
            self.distorch_allocations = {}
            self.distorch_compute_device = kwargs.get("compute_device", None)
            if self.distorch_compute_device is not None:
                current_device = self.distorch_compute_device

            register_patched_ggufmodelpatcher(self)

            for key, value in list(kwargs.items()):
                if key not in {"unet_name", "clip_name1", "clip_name2", "clip_name2", "type"}:
                    logging.info(f"MultiGPU: Removing {key} from kwargs")
                    logging.info(f"MultiGPU: Value: {value}")
                    self.distorch_allocations[key] = kwargs.pop(key)

            fn = getattr(super(), cls.FUNCTION)
            model = fn(*args, **kwargs)

            if hasattr(model[0], 'model'):
                model_hash = create_model_hash(model[0], "override")
                model_allocation_store[model_hash] = self.distorch_allocations.copy()
                logging.info(f"MultiGPU: STORE - Hash {model_hash}, Allocations: {model_allocation_store[model_hash]}")
            elif hasattr(model[0], 'patcher') and hasattr(model[0].patcher, 'model'):
                model_hash = create_model_hash(model[0].patcher, "override") 
                model_allocation_store[model_hash] = self.distorch_allocations.copy()
                logging.info(f"MultiGPU: STORE - Hash {model_hash}, Allocations: {model_allocation_store[model_hash]}")
            return model

    return NodeOverrideDisTorch