# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoMimicV3 is a 1.3B parameter audio-driven human animation model that generates videos from a single reference image and audio input. It's based on the Wan2.1 diffusion architecture and supports both "preview" (full quality) and "flash" (fast, 8-step) inference modes.

## Architecture

The model uses a diffusion transformer architecture with these key components:
- **Transformer**: `src/wan_transformer3d_audio.py` - Core 3D transformer with audio conditioning
- **VAE**: `src/wan_vae.py` - Autoencoder for video encoding/decoding (temporal compression 4x, spatial 8x)
- **Text Encoder**: `src/wan_text_encoder.py` - T5-based encoder (4096 dim, 24 layers)
- **Image Encoder**: `src/wan_image_encoder.py` - CLIP model for reference image encoding
- **Audio Encoder**: Wav2Vec2 (facebook/wav2vec2-base-960h for preview, chinese-wav2vec2-base for flash)
- **Pipeline**: `src/pipeline_wan_fun_inpaint_audio.py` - Main diffusion pipeline combining all components

## Entry Points

- `app_mm.py` - Gradio web UI for preview model (12G VRAM, 768x768)
- `infer_preview.py` - CLI inference for preview model
- `infer_flash.py` - CLI inference for flash model (8-step, no face mask required)

## Running the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio UI (preview model)
python app_mm.py --server_name 0.0.0.0 --server_port 7891 --share

# Run CLI inference (preview)
python infer_preview.py

# Run flash inference
bash run_flash.sh
```

## Key Inference Parameters

- `guidance_scale`: 3.0-6.0 (text following), default 4.5
- `audio_guidance_scale`: 1.8-3.0 (lip sync quality), default 2.5-2.9
- `teacache_threshold`: 0-0.1 (speed optimization)
- `partial_video_length`: 81 for 16G VRAM, 113 for 24G VRAM, 49 for 12G VRAM
- `num_inference_steps`: 5 for talking head, 15-25 for talking body

## Key Implementation Details

1. **Video chunking**: Long videos are generated in overlapping chunks with blend transitions at overlap boundaries
2. **Face masking**: IP mask identifies the face region for localized animation
3. **TeaCache**: Enables skip of similar computation steps based on learned coefficients
4. **Dynamic CFG**: Dynamic guidance scale adjustment during sampling
5. **Multi-GPU support**: Ulysses (sequence) and Ring (attention) parallelism via `src/dist.py`

## Model File Structure

Expected structure for preview model:
```
./preview/
├── Wan2.1-Fun-V1.1-1.3B-InP/    # Base Wan2.1 model
├── wav2vec2-base-960h/           # Audio encoder
└── transformer/
    └── diffusion_pytorch_model.safetensors
```

Flash model uses `chinese-wav2vec2-base` instead and `config/wan2.1/wan_civitai.yaml`.