```markdown
# ComfyUI-MultiGPU

### Experimental nodes for using multiple GPUs in a single ComfyUI workflow.

This extension adds device selection capabilities to model loading nodes in ComfyUI. It monkey patches the memory management of ComfyUI in a hacky way and is neither a comprehensive solution nor a well-tested one. Use at your own risk.

Note that this does not add parallelism. The workflow steps are still executed sequentially just on different GPUs. Any potential speedup comes from not having to constantly load and unload models from VRAM.

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-MultiGPU` in the list of nodes and follow installation instructions.

## Manual Installation

Clone [this repository](https://github.com/pollockjj/ComfyUI-MultiGPU) inside `ComfyUI/custom_nodes/`.

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

- XLabAI FLUX ControlNet:
  - LoadFluxControlNetMultiGPU (requires [x-flux-comfy](https://github.com/XLabAI/x-flux-comfyui))

- Florence2:
  - Florence2ModelLoaderMultiGPU (requires [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2))
  - DownloadAndLoadFlorence2ModelMultiGPU

- LTX Video Custom Checkpoint Loader:
  - LTXVLoaderMultiGPU (requires [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo))

All MultiGPU nodes can be found in the "multigpu" category in the node menu.

## Example workflows

All workflows have been tested on a 2x 3090 setup.

### Loading two SDXL checkpoints on different GPUs

- [examples/sdxl_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/sdxl_2gpu.json)

This workflow loads two SDXL checkpoints on two different GPUs. The first checkpoint is loaded on GPU 0, and the second checkpoint is loaded on GPU 1.

### Split FLUX.1-dev across two GPUs

- [examples/flux1dev_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_2gpu.json)

This workflow loads a FLUX.1-dev model and splits it across two GPUs. The UNet model is loaded on GPU 1 while the text encoders and VAE are loaded on GPU 0.

### FLUX.1-dev and SDXL in the same workflow

- [examples/flux1dev_sdxl_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_sdxl_2gpu.json)

This workflow loads a FLUX.1-dev model and an SDXL model in the same workflow. The FLUX.1-dev model has its UNet on GPU 1 with VAE and text encoders on GPU 0, while the SDXL model uses separate allocations.

### Using GGUF quantized models across GPUs

- [examples/flux1dev_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_2gpu_GGUF.json)

This workflow demonstrates using quantized GGUF models split across multiple GPUs for reduced VRAM usage with the UNet on GPU 1, VAE and text encoders on GPU 0.

### EXPERIMENTAL - USE AT YOUR OWN RISK

These workflows combine multiple features and non-core loaders types and may require significant VRAM to execute. They are provided as examples of what's possible but may require adjustment for your specific setup.

#### Image to Prompt to Image to Video Generation Pipeline

- [examples/florence2_flux1dev_ltxv_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/florence2_flux1dev_ltxv_2gpu_GGUF.json)

This workflow creates an img2txt2img2vid video generation pipeline by:
1. Providing a starting image for analysis by Florence2
2. Using the Florence2 data for a FLUX.1 Dev image prompt
3. Taking the resulting FLUX.1 image and provide it as the starting image for an LTX Video image-to-video generation
4. Generate a 5 second video based on the provided image
All models are distributed across available GPUs with no reloading on dual 3090s

#### LLM-Guided Video Generation

- [examples/llamacpp_ltxv_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/llamacpp_ltxv_2gpu_GGUF.json)

This workflow demonstrates:
1. Using a local LLM (loaded on first GPU via llama.cpp) to take a text suggestion and craft an LTX Video promot
2. Feeding the enhanced prompt to LTXVideo (loaded on second GPU) for video generation
Requires appropriate LLM and LTXVideo models.

## Support

If you encounter problems, please [open an issue](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/new). Attach the workflow if possible.

## Credits

Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov). 
Implementation improved by [City96](https://v100s.net/). 
Currently maintained by [pollockjj](https://github.com/pollockjj).