# ComfyUI-MultiGPU

## Advanced user nodes for using multiple GPUs as well as offloading model components to the CPU in a single ComfyUI workflow

This extension adds device selection capabilities to model loading nodes in ComfyUI. It monkey patches the memory management of ComfyUI in a hacky way and is neither a comprehensive solution is nor is it well-tested on any edge-case CUDA/CPU solutions. **Use at your own risk.**

*Note: This does not add parallelism. The workflow steps are still executed sequentially just with model components loaded on different GPUs or offloaded to the CPU where allowed. Any potential speedup comes from not having to constantly load and unload models from VRAM.*

# NEW: DisTorch - Advanced GGUF-Quantized Model Layer Distribution

DisTorch nodes are now available, allowing fine-grained control over model layer distribution across multiple devices for GGUF quantized models. Using a simple allocation string (e.g., "cuda:0,0.025;cuda:1,0.05;cpu,0.10"), you can precisely specify how much memory each device should contribute to hosting model layers. This enables sophisticated memory management strategies like:

- Splitting large models across multiple GPUs with different VRAM capacities
- Utilizing CPU memory alongside GPU VRAM for handling memory-intensive models
- Optimizing layer placement based on your specific hardware configuration

Check out the updated examples `hunyuan_gguf_distorch.json` and `flux1dev_gguf_distorch.json` to see DisTorch in action, demonstrating advanced layer distribution across multiple devices.

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
- NF4 Checkpoint Format Loader(requires [ComfyUI_bitsandbytes_NF4](https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4)):
  - CheckpointLoaderNF4MultiGPU
- HunyuanVideoWrapper (requires [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)):
  - HyVideoModelLoaderMultiGPU
  - HyVideoModelLoaderDiffSynthMultiGPU (**NEW** - MultiGPU-specific node for offloading to an `offload_device` using MultiGPU's device selectors)
  - HyVideoVAELoaderMultiGPU
  - DownloadAndLoadHyVideoTextEncoderMultiGPU
- Native to ComfyUI-MultiGPU
  - DeviceSelectorMultiGPU (Allows user to link loaders together to use the same selected device)

All MultiGPU nodes available for your install can be found in the "multigpu" category in the node menu.

## Example workflows

All workflows have been tested on a 2x 3090 linux setup, a 4070 win 11 setup, and a 3090/1070ti linux setup.

### Split GGUF-quantized UNet and CLIP models across multiple devices using DisTorch

- [examples/hunyuan_gguf_distorch.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuan_gguf_distorch.json)
This workflow attaches a HunyuanVideo GGUF-quantized model on `cuda:0` for compute and distrubutes its UNet across itself, a secondary CUDA device, and the system's main memory (`cpu`) using a new DisTorch distributed-load methodology. The text encoder now attaches itself to `cuda:1` and splits iteself between `cuda:1` amd `cpu` layers. While the VAE is loaded on GPU 1 directly and use `cuda:1` for compute.

- [examples/flux1dev_gguf_distorch.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/flux1dev_gguf_distorch.json)
This workflow loads a FLUX.1-dev model on `cuda:0` for compute and distrubutes its UNet across multiple CUDA devices using new DisTorch distributed-load methodology. While the text encoders and VAE are loaded on GPU 1 and use `cuda:1` for compute.

### Split Hunyuan Video UNet across two devices and use DiffSynth Just-in-Time loading

- [examples/hunyuanvideowrapper_diffsynth.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuanvideowrapper_diffsynth.json)
This workflow demonstrates DiffSynth's memory optimization strategy enabled in kijai's `ComfyUI-HunyuanVideoWrapper` UNet loader, splitting the UNet model across two CUDA devices using block-swapping. The main device handles active computations while blocks are swapped to and from the offload device as needed. As written, the CLIP loads on cuda:0 and then offloads, and the VAE is loaded after the UNet model has been cleared from memory after generation. This approach enables processing of higher resolution or longer duration videos that would exceed a single GPU's memory capacity, though at the cost of additional processing time. Note that an initial OOM error is expected as the workflow calibrates its memory management strategy - simply run the generation again with the same parameters.

### Split Hunyuan Video generation across multiple resources

- [examples/hunyuanvideowrapper_native_vae.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuanvideowrapper_native_vae.json)
This workflow uses two of the custom_node's loaders - putting the main video model on cuda0 and the CLIP onto the CPU. This workflow uses Comfy's native VAE loader to load the VAE onto a second cuda device, keeping model, VAE, and CLIP in their operating memory space at all times. This allows the benefit of kijai's proecessing node with the flexibility of a MultiGPU setup.

- [examples/hunyuanvideowrapper_select_device.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuanvideowrapper_select_device.json)
This workflow loads the main video model and VAE onto the cuda device and the CLIP onto the cpu. The model and VAE are linked in this example due to kijai's own extensive memory management assuming model and VAE are on the same device.

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

### Using DeviceSelectorMultiGPU

- [examples/device_selector_lowvram_flux_controlnet.json](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/device_selector_lowvram_flux_controlnet.json)
This workflow loads a GGUF version of FLUX.1-dev onto a cuda device and the T5 CLIP onto the CPU. The FLUX.1-dev fill controlnet model by alimama-creative [FLUX.1-dev-Controlnet-Inpainting-Alpha controlnet model](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/tree/main) illustrating linking two loaders together so their resources always remain in-sync.

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
