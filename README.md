# Piczie: Fast Open Text-to-Image

Piczie is a Streamlit-based application for generating images from text prompts using open diffusion models such as **Stable Diffusion Turbo** and **FLUX.1**.  
It provides a simple web interface while handling model loading, memory optimizations, and gated-model authentication with Hugging Face.
[Piczel](https://piczel.streamlit.app)

---

## Features
- **Multiple Models**  
  - [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) for fast and lightweight image generation.  
  - [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) for higher quality images (requires Hugging Face access approval).  

- **Smart Defaults**  
  Automatically adjusts inference steps, guidance scale, and image size based on the chosen model and available hardware.

- **CPU and GPU Support**  
  Runs on CPU (slower, limited resolution) or NVIDIA GPU (faster, higher quality).  

- **Memory Optimizations**  
  Uses attention slicing and optional sequential CPU offload to reduce VRAM and RAM usage.

- **Streamlit UI**  
  Lightweight, browser-based interface for easy interaction.

---

## Requirements

- Python 3.9+
- Hugging Face account with access tokens for gated models (FLUX family)
- Dependencies:
  ```txt
  streamlit
  torch
  diffusers>=0.30.0
  transformers
  accelerate
  safetensors
  Pillow
  huggingface_hub
  xformers     # optional, for faster GPU inference