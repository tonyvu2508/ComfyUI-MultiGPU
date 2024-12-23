# ComfyUI-MultiGPU

### Experimental nodes for using multiple GPUs in a single ComfyUI workflow.

This extension adds device selection capabilities to model loading nodes in ComfyUI. It monkey patches the memory management of ComfyUI in a hacky way and is neither a comprehensive solution nor a well-tested one. Use at your own risk.

Note that this does not add parallelism. The workflow steps are still executed sequentially just on different GPUs. Any potential speedup comes from not having to constantly load and unload models from VRAM.

## Installation

Clone this repository inside `ComfyUI/custom_nodes/`.

## Nodes

The extension automatically creates MultiGPU versions of loader nodes. Each MultiGPU node has the same functionality as its original counterpart but adds a `device` parameter that allows you to specify the GPU to use.

Currently supported nodes (automatically detected if available):
- Standard ComfyUI loaders:
  - CheckpointLoaderSimpleMultiGPU
  - CLIPLoaderMultiGPU
  - ControlNetLoaderMultiGPU 
  - DualCLIPLoaderMultiGPU
  - TripleCLIPLoaderMultiGPU
  - UNETLoaderMultiGPU
  - VAELoaderMultiGPU

- GGUF loaders (requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)):
  - UnetLoaderGGUFMultiGPU (supports quantized models like [flux1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf))
  - UnetLoaderGGUFAdvancedMultiGPU 
  - CLIPLoaderGGUFMultiGPU
  - DualCLIPLoaderGGUFMultiGPU
  - TripleCLIPLoaderGGUFMultiGPU

- Additional supported nodes:
  - LoadFluxControlNet (requires [x-flux-comfy](https://github.com/XLabAI/x-flux-comfyui))

All MultiGPU nodes can be found in the "multigpu" category in the node menu.

## Example workflows

All workflows have been tested on a 2x 3090 setup.

### Loading two SDXL checkpoints on different GPUs

- [Download](examples/sdxl_2gpu.json)

This workflow loads two SDXL checkpoints on two different GPUs. The first checkpoint is loaded on GPU 0, and the second checkpoint is loaded on GPU 1.

### Split FLUX.1-dev across two GPUs

- [Download](examples/flux1dev_2gpu.json)

This workflow loads a FLUX.1-dev model and splits it across two GPUs. The UNet model is loaded on GPU 0 while the text encoders and VAE are loaded on GPU 1.

### FLUX.1-dev and SDXL in the same workflow

- [Download](examples/flux1dev_sdxl_2gpu.json)

This workflow loads a FLUX.1-dev model and an SDXL model in the same workflow. The FLUX.1-dev model is loaded on GPU 0, and the SDXL model is loaded on GPU 1.

## Support

If you encounter problems, please [open an issue](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/new). Attach the workflow if possible.

## Credits

Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov).  
Implementation improved by [City96](https://v100s.net/).  
Currently maintained by [pollockjj](https://github.com/pollockjj).