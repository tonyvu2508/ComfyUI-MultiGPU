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
 

# Register desired nodes
register_module("",                         ["UNETLoader", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "CheckpointLoaderSimple", "ControlNetLoader"])

register_LTXVLoaderMultiGPU()
register_Florence2ModelLoaderMultiGPU()
register_DownloadAndLoadFlorence2ModelMultiGPU()
register_CheckpointLoaderNF4()
register_LoadFluxControlNetMultiGPU()
register_MMAudioModelLoaderMultiGPU()
register_MMAudioFeatureUtilsLoaderMultiGPU()
register_MMAudioSamplerMultiGPU()
register_UnetLoaderGGUFMultiGPU()
register_CLIPLoaderGGUFMultiGPU()


logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
