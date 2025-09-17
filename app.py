# app.py
import io
import os
from functools import lru_cache

import streamlit as st
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

# Try to import FluxPipeline (required for FLUX models on newer diffusers)
try:
    from diffusers import FluxPipeline  # type: ignore
    HAS_FLUX = True
except Exception:
    HAS_FLUX = False

# ---------- Config ----------
st.set_page_config(page_title="Fast Open Text-to-Image", page_icon="⚡", layout="wide")

MODEL_CHOICES = [
    "black-forest-labs/FLUX.1-schnell",  # fastest
    "black-forest-labs/FLUX.1-dev",      # higher quality
    "stabilityai/sd-turbo",              # very fast baseline
]

DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

# ---------- Helpers ----------
def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32

@lru_cache(maxsize=3)
def load_pipeline(model_id: str):
    device, dtype = device_and_dtype()

    if "black-forest-labs/FLUX" in model_id:
        if not HAS_FLUX:
            raise RuntimeError(
                "FluxPipeline not found in your diffusers version. "
                "Upgrade: pip install -U diffusers transformers accelerate"
            )
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=(torch.bfloat16 if dtype == torch.bfloat16 else torch.float16),
            use_safetensors=True,
        )
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe, device

def smart_defaults(model_id, steps, guidance):
    if "FLUX.1-schnell" in model_id:
        return (steps or 4, guidance if guidance is not None else 0.0)
    if "FLUX.1-dev" in model_id:
        return (steps or 28, guidance if guidance is not None else 3.5)
    if "sd-turbo" in model_id:
        return (steps or 2, guidance if guidance is not None else 0.0)
    return (steps or 30, guidance if guidance is not None else 7.0)

def to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ---------- UI ----------
st.markdown("# ⚡ Fast Open Text-to-Image")
st.caption("FLUX (schnell/dev) and SD-Turbo via 🤗 diffusers. Free to run locally.")

with st.sidebar:
    st.subheader("Settings")
    model_id = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
    width = st.slider("Width", 256, 1024, 768, step=64)
    height = st.slider("Height", 256, 1024, 768, step=64)
    steps = st.slider("Steps (0 = smart default)", 0, 60, 0, step=1)
    guidance = st.number_input("Guidance (CFG; blank=smart)", value=None, step=0.1, placeholder="auto")
    seed_in = st.text_input("Seed (blank = random)", value="")
    gen_btn = st.button("Generate", type="primary", use_container_width=True)

prompt = st.text_area(
    "Prompt",
    placeholder="An ultra-detailed cinematic photo of a snow-covered cabin at sunrise",
    height=80,
)

# ---------- Generate ----------
if gen_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    try:
        pipe, device = load_pipeline(model_id)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    steps, guidance = smart_defaults(model_id, steps, guidance)

    if seed_in.strip() == "":
        generator = None
    else:
        try:
            seed_val = int(seed_in)
            generator = torch.Generator(device=device).manual_seed(seed_val)
        except ValueError:
            st.warning("Seed must be an integer; using random seed.")
            generator = None

    run_kwargs = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=generator,
    )

    with st.spinner("Generating…"):
        if device == "cuda":
            with torch.autocast("cuda"):
                out = pipe(**run_kwargs)
        else:
            out = pipe(**run_kwargs)

    image = out.images[0]
    st.image(image, caption=f"{model_id} | steps={steps} cfg={guidance}", use_column_width=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download PNG",
            data=to_bytes(image, "PNG"),
            file_name="t2i_output.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "Download JPG",
            data=to_bytes(image.convert("RGB"), "JPEG"),
            file_name="t2i_output.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

# ---------- Footer ----------
dev, dtype = device_and_dtype()
st.sidebar.caption(f"Device: **{dev.upper()}**, DType: **{str(dtype).split('.')[-1]}**")
st.sidebar.caption("Tip: For CPU, try 512×512 and fewer steps.")
