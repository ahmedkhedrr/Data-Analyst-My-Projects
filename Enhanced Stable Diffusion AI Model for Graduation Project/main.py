
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from fpdf import FPDF
import torch
import requests
from transformers import pipeline
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image, ImageFilter
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure font file exists
FONT_PATH = "DejaVuSans.ttf"
if not Path(FONT_PATH).exists():
    logging.warning(f"Font file {FONT_PATH} not found. Downloading...")
    try:
        response = requests.get("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf")
        with open(FONT_PATH, "wb") as f:
            f.write(response.content)
        logging.info("Font downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download font: {e}")
        raise

def create_prompt(base_prompt, style="photorealistic"):
    """Create an enhanced prompt with a specified style."""
    style_presets = {
        "photorealistic": "photorealistic, cinematic lighting, 4k resolution, intricate details, vibrant colors, sharp focus",
        "painting": "oil painting, vibrant colors, impressionist style, textured brushstrokes",
        "3d": "3d render, hyper-realistic, octane render style, smooth gradients",
        "sketch": "pencil sketch, detailed linework, monochromatic, high contrast"
    }
    if style not in style_presets:
        logging.warning(f"Style '{style}' not found. Using default 'photorealistic'.")
        style = "photorealistic"
    return f"A {style_presets[style]} depiction of {base_prompt}"

def generate_image(
    prompt,
    output_path="output.png",
    resolution=(768, 768),
    device="cuda" if torch.cuda.is_available() else "cpu",
    inference_steps=75,
    guidance_scale=8.0,
    seed=None,
    scheduler_type="dpm++_2m_k",
    use_lora=False,
    lora_path=None,
    style="photorealistic",
    upscale=False
):
    """Generate an enhanced image using Stable Diffusion v1.5."""
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        logging.info(f"Running Stable Diffusion on device: {device}")

        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("xFormers enabled for memory-efficient attention.")
        except:
            logging.warning("xFormers not available, falling back to default attention.")

        if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
            pipe.enable_sequential_cpu_offload()
            logging.info("Enabled sequential CPU offloading for low VRAM.")

        if use_lora and lora_path:
            pipe.load_lora_weights(lora_path)
            logging.info(f"Loaded LoRA weights from {lora_path}")

        scheduler_map = {
            "dpm++_2m_k": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++"),
            "euler_a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        }
        if scheduler_type in scheduler_map:
            pipe.scheduler = scheduler_map[scheduler_type]
            logging.info(f"Using scheduler: {scheduler_type}")

        enhanced_prompt = create_prompt(prompt, style)
        negative_prompt = (
            "blurry, low quality, artifacts, distorted, text, watermark, low resolution, "
            "overexposed, underexposed, noise, grainy, unrealistic"
        )
        logging.info(f"Enhanced prompt: {enhanced_prompt}")

        if seed is not None:
            torch.manual_seed(seed)
            logging.info(f"Using seed: {seed}")

        logging.info("Generating image...")
        image = pipe(
            enhanced_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            height=resolution[0],
            width=resolution[1]
        ).images[0]

        # Post-processing: Sharpen and optional upscale
        image = image.filter(ImageFilter.SHARPEN)
        if upscale:
            try:
                from realesrgan import RealESRGAN
                upscaler = RealESRGAN(device=device, scale=2)
                image_np = np.array(image)
                image_np = upscaler.enhance(image_np)[0]
                image = Image.fromarray(image_np)
                logging.info("Image upscaled using Real-ESRGAN.")
            except ImportError:
                logging.warning("RealESRGAN not installed. Skipping upscaling.")

        image.save(output_path, quality=95)
        logging.info(f"Image saved to {output_path}")

        return image
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        raise

def generate_text(prompt, max_length=500, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Generate text using Flan-T5-Base for scientific explanations."""
    try:
        pipe = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if device == "cuda" else -1)
        
        text_prompt = f"Explain this in two paragraphs: {prompt}"
        
        logging.info("Generating text...")
        generated_text = pipe(
            text_prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3
        )[0]['generated_text']
        
        return generated_text
    except Exception as e:
        logging.error(f"Text generation failed: {e}")
        return "Failed to generate description."

def create_pdf(prompt, text, image_path, output_pdf="imagedocu2.pdf"):
    """Create a PDF with text and image."""
    class PDF(FPDF):
        def header(self):
            self.set_font("DejaVu", "", 12)
            self.cell(0, 10, "AI-Generated Educational Content", ln=True, align="C")

        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", "", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    try:
        pdf = PDF()
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.set_font("DejaVu", "", 12)
        pdf.add_page()

        pdf.set_font("DejaVu", "", 14)
        pdf.cell(0, 10, "Prompt:", ln=True)
        pdf.set_font("DejaVu", "", 12)
        pdf.multi_cell(0, 10, prompt)

        pdf.ln(5)
        pdf.set_font("DejaVu", "", 14)
        pdf.cell(0, 10, "Description:", ln=True)
        pdf.set_font("DejaVu", "", 12)
        pdf.multi_cell(0, 10, text)

        if pdf.get_y() > 200:
            pdf.add_page()
        img_width = min(190, pdf.w - 20)
        pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=img_width)

        pdf.output(output_pdf)
        logging.info(f"PDF created: {output_pdf}")
    except Exception as e:
        logging.error(f"PDF creation failed: {e}")
        raise

def main():
    prompt = "Generate for me a futuristic image of the pyramids with high quality in details, vibrant colors and modern design"
    style = "3d"  # Options: photorealistic, painting, 3d, sketch
    
    image = generate_image(
        prompt,
        resolution=(768, 768),
        inference_steps=75,
        guidance_scale=8.0,
        seed=42,
        scheduler_type="dpm++_2m_k",
        style=style,
        upscale=False
    )
    
    text = generate_text(prompt, max_length=500)
    
    create_pdf(prompt, text, "output.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
