# ComfyUI-MultiGPU

## Experimental nodes for using multiple GPUs as well as offloading model components to the CPU in a single ComfyUI workflow

This extension adds device selection capabilities to model loading nodes in ComfyUI. It monkey patches the memory management of ComfyUI in a hacky way and is neither a comprehensive solution is nor is it well-tested on any edge-case CUDA/CPU solutions. **Use at your own risk.**

*Note: This does not add parallelism. The workflow steps are still executed sequentially just with model components loaded on different GPUs or offloaded to the CPU where allowed. Any potential speedup comes from not having to constantly load and unload models from VRAM.*

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-MultiGPU` in the list of nodes and follow installation instructions.

## Manual Installation

Clone [this repository](https://github.com/pollockjj/ComfyUI-MultiGPU) inside `ComfyUI/custom_nodes/`.

## Nodes

The extension automatically creates MultiGPU versions of loader nodes. Each MultiGPU node has the same functionality as its original counterpart but adds a `device` parameter that allows you to specify the GPU to use.

Currently supported nodes (automatically detected if available):

- Standard [ComfyUI](https://github.com/comfyanonymous/ComfyUI) model loaders:
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
- XLabAI FLUX ControlNet (requires [x-flux-comfy](https://github.com/XLabAI/x-flux-comfyui)):
  - LoadFluxControlNetMultiGPU
- Florence2 (requires [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)):
  - Florence2ModelLoaderMultiGPU
  - DownloadAndLoadFlorence2ModelMultiGPU
- LTX Video Custom Checkpoint Loader (requires [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)):
  - LTXVLoaderMultiGPU
- LTX Video Custom Checkpoint Loader (requires [ComfyUI_bitsandbytes_NF4](https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4)):
  - CheckpointLoaderNF4MultiGPU

All MultiGPU nodes available for your install can be found in the "multigpu" category in the node menu.

## Example workflows

All workflows have been tested on a 2x 3090 setup.

### Split FLUX.1-dev across two GPUs

- [examples/flux1dev_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_2gpu.json)
This workflow loads a FLUX.1-dev model and splits its components across two GPUs. The UNet model is loaded on GPU 1 while the text encoders and VAE are loaded on GPU 0.

### Split FLUX.1-dev between the CPU and a single GPU

- [examples/flux1dev_cpu_1gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_cpu_1gpu_GGUF.json)
This workflow demonstrates splitting a quantized, GGUF FLUX.1-dev model between a CPU and a single GPU. The UNet model is loaded on the GPU, while the VAE and text encoders are handled by the CPU. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

### Using GGUF quantized models across GPUs

- [examples/flux1dev_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_2gpu_GGUF.json)
This workflow demonstrates using quantized GGUF models split across multiple GPUs for reduced VRAM usage with the UNet on GPU 1, VAE and text encoders on GPU 0. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

### Using GGUF quantized models across a CPU and a single GPU for video generation

- [examples/hunyuan_cpu_1gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuan_cpu_1gpu_GGUF.json)
This workflow demonstrates using quantized GGUF models for Hunyan Video split across the CPU and one GPU. In this instance, a quantized video model's UNet and VAE are on GPU 0, whereas a split of one standard and one GGUF model text encoder are on the CPU. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

### Using GGUF quantized models across GPUs for video generation

- [examples/hunyuan_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuan_2gpu_GGUF.json)
This workflow demonstrates using quantized GGUF models for Hunyan Video split across multiple GPUs. In this instance, a quantized video model's UNet is on GPU 0 whereas the VAE and text encoders are on GPU 1. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

### Loading two SDXL checkpoints on different GPUs

- [examples/sdxl_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/sdxl_2gpu.json)
This workflow loads two SDXL checkpoints on two different GPUs. The first checkpoint is loaded on GPU 0, and the second checkpoint is loaded on GPU 1.

### FLUX.1-dev and SDXL in the same workflow

- [examples/flux1dev_sdxl_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_sdxl_2gpu.json)
This workflow loads a FLUX.1-dev model and an SDXL model in the same workflow. The FLUX.1-dev model has its UNet on GPU 1 with VAE and text encoders on GPU 0, while the SDXL model uses separate allocations on GPU 0.

### Image to Prompt to Image to Video Generation Pipeline

- [examples/florence2_flux1dev_ltxv_cpu_2gpu.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/florence2_flux1dev_ltxv_cpu_2gpu.json)
This workflow creates an img2txt2img2vid video generation pipeline by:

 1. Loading the Florence2 model on the CPU and providing a starting image for analysis and generating a text response
 2. Loading FLUX.1 Dev UNET on GPU 1, with CLIP and VAE on the CPU and generating an image using the Florence2 text as a prompt
 3. Loading the LTX Video UNet and VAE on GPU 2, and LTX-encoded CLIP on the CPU, and taking the resulting FLUX.1 image and provide it as the starting image for an LTX Video image-to-video generation
 4. Generate a 5 second video based on the provided image
All models are distributed across available the available CPU and GPUs with no model reloading on dual 3090s. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)

#### LLM-Guided Video Generation

- [examples/llamacpp_ltxv_2gpu_GGUF.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/llamacpp_ltxv_2gpu_GGUF.json)
This workflow demonstrates:

1. Using a local LLM (loaded on first GPU via llama.cpp) to take a text suggestion and craft an LTX Video promot
2. Feeding the enhanced prompt to LTXVideo (loaded on second GPU) for video generation
Requires appropriate LLM. Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

## Support

If you encounter problems, please [open an issue](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/new). Attach the workflow if possible.

## Credits

Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov).
Implementation improved by [City96](https://v100s.net/).
Currently maintained by [pollockjj](https://github.com/pollockjj).
