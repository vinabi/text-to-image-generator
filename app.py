# app.py
import io
import os
import gc
from functools import lru_cache

import streamlit as st
import torch
from PIL import Image
from huggingface_hub import login
from diffusers import AutoPipelineForText2Image

# FLUX pipeline (may require newer diffusers)
try:
    from diffusers import FluxPipeline  # type: ignore
    HAS_FLUX = True
except Exception:
    HAS_FLUX = False

# ---- Page config ----
st.set_page_config(page_title="piczie", page_icon="⚡", layout="wide")

# ---- Hugging Face token (secrets → env fallback) ----
HF_TOKEN = None
try:
    HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)
except Exception:
    pass
HF_TOKEN = HF_TOKEN or os.environ.get("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        st.warning(f"HF login failed: {e}")

# ---- Models ----
MODEL_SMALL = "stabilityai/sd-turbo"  # safest everywhere
MODEL_FLUX_FAST = "black-forest-labs/FLUX.1-schnell"  # gated, heavy
MODEL_FLUX_QUAL = "black-forest-labs/FLUX.1-dev"      # gated, heavier

MODEL_CHOICES = [MODEL_FLUX_FAST, MODEL_FLUX_QUAL, MODEL_SMALL]
DEFAULT_MODEL = MODEL_FLUX_FAST

# ---- Helpers ----
def has_cuda() -> bool:
    return torch.cuda.is_available()

def device_and_dtype():
    if has_cuda():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32

def is_gated(model_id: str) -> bool:
    return "black-forest-labs/FLUX" in model_id

def require_token_if_gated(model_id: str):
    if is_gated(model_id) and not HF_TOKEN:
        raise RuntimeError(
            "This is a gated model. Add HUGGINGFACE_TOKEN in Streamlit Secrets (or env) to use FLUX, "
            "or switch to 'stabilityai/sd-turbo'."
        )

def smart_defaults(model_id, width, height, steps, guidance, device):
    # Smaller & safer on CPU
    if device == "cpu":
        width = min(width or 512, 512)
        height = min(height or 512, 512)
    if MODEL_FLUX_FAST in model_id:
        return width or 768, height or 768, steps or 4, (0.0 if guidance is None else guidance)
    if MODEL_FLUX_QUAL in model_id:
        return width or 768, height or 768, steps or 24, (3.0 if guidance is None else guidance)
    if MODEL_SMALL in model_id:
        return width or 512, height or 512, steps or 2, (0.0 if guidance is None else guidance)
    return width or 768, height or 768, steps or 30, (7.0 if guidance is None else guidance)

@lru_cache(maxsize=3)
def load_pipeline(requested_model: str):
    # Force sd-turbo when no GPU (FLUX tends to crash on CPU/low RAM)
    model_id = requested_model
    if not has_cuda() and requested_model != MODEL_SMALL:
        model_id = MODEL_SMALL

    require_token_if_gated(model_id)

    device, dtype = device_and_dtype()

    if "black-forest-labs/FLUX" in model_id:
        if not HAS_FLUX:
            raise RuntimeError("Update diffusers for FluxPipeline, or pick 'stabilityai/sd-turbo'.")
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
    else:
        # reduce CPU thread pressure a bit
        try:
            torch.set_num_threads(max(1, torch.get_num_threads() // 2))
        except Exception:
            pass

    pipe.enable_attention_slicing()
    try:
        pipe.enable_sequential_cpu_offload()  # big saver on tight VRAM
    except Exception:
        pass

    pipe = pipe.to(device)
    return pipe, device, model_id  # return effective model (may be forced to sd-turbo)

def safe_generate(pipe, device, **kwargs):
    try:
        if device == "cuda":
            with torch.autocast("cuda"):
                out = pipe(**kwargs)
        else:
            out = pipe(**kwargs)
        return out.images[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise RuntimeError("Out of memory. Try 'stabilityai/sd-turbo', 512×512, and fewer steps.") from e
        raise

def to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ---- UI ----
st.markdown("# Piczie")
st.caption("FLUX (schnell/dev) and SD-Turbo via 🤗 diffusers. Handles gated models with your HF token.")

with st.sidebar:
    st.subheader("Settings")
    model_pick = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(DEFAULT_MODEL))
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

# ---- Generate ----
if gen_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    try:
        pipe, device, effective_model = load_pipeline(model_pick)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    width, height, steps, guidance = smart_defaults(effective_model, width, height, steps, guidance, device)

    # Seed handling
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
        try:
            image = safe_generate(pipe, device, **run_kwargs)
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.image(image, caption=f"{effective_model} | steps={steps} cfg={guidance} | {device.upper()}", use_column_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download PNG", to_bytes(image, "PNG"), "t2i_output.png", "image/png", use_container_width=True)
    with c2:
        st.download_button("Download JPG", to_bytes(image.convert("RGB"), "JPEG"), "t2i_output.jpg", "image/jpeg", use_container_width=True)

# ---- Footer ----
dev, dtype = device_and_dtype()
hf_status = "token loaded" if HF_TOKEN else "no token (FLUX gated)"
st.sidebar.caption(f"Device: **{dev.upper()}**, DType: **{str(dtype).split('.')[-1]}**")
st.sidebar.caption(f"Hugging Face: {hf_status}")
st.sidebar.caption("Tip: On CPU use sd-turbo, 512×512, steps 2.")
