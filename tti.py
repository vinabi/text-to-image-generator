import gradio as gr
from diffusers import DiffusionPipeline
import torch

def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype = torch.float16, use_safetensors = True, variant = "fp16")
    pipe = pipe.to('cuda')  
    return pipe

try:
    pipe = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def generate_image(prompt):
    try:
        result = pipe(prompt)
        image = result.images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

prompt_input = gr.Textbox(label = "Enter your prompt", placeholder = "Type something here...", lines = 1)
generate_button = gr.Button("Generate Image")
image_output = gr.Image(label = "Generated Image")

interface = gr.Interface
(
    fn = generate_image,
    inputs = prompt_input,
    outputs = image_output,
    live = False,
    title = "Text-to-Image Generator",
    description = "Generate images from text prompts using Stable Diffusion.",
    theme = "default"  
)

interface.launch()
