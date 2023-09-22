import sys
import time
import torch
from io import BytesIO
from pydantic import BaseModel
import base64

torch.backends.cuda.matmul.allow_tf32 = True

import modal
from modal import web_endpoint
from ..common import stub 

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.save_pretrained(cache_path+"/pipe", safe_serialization=True)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.save_pretrained(cache_path+"/refiner", safe_serialization=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "xformers"
    )
    .run_function(download_models)
)

class Item(BaseModel):
    prompt: str = None

# Code Adapted from https://github.com/basetenlabs/truss-examples/blob/main/stable-diffusion-xl-1.0
@stub.cls(gpu="A100", image=image, timeout=180)
class Image:
    def __enter__(self):
        from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler
        self.pipe = DiffusionPipeline.from_pretrained(cache_path+"/pipe", torch_dtype=torch.float16)
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.to(device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.refiner = DiffusionPipeline.from_pretrained(cache_path+"/refiner", torch_dtype=torch.float16)
        self.refiner.to(device)
        self.refiner.enable_xformers_memory_efficient_attention()

    @web_endpoint(method="POST")
    def api(self, item: Item):
        try:
            start = time.time()
            image = self.pipe(prompt=item.prompt,
                          negative_prompt=None,
                          num_inference_steps=20,
                          denoising_end=0.8,
                          guidance_scale=7.5,
                          output_type="latent").images[0]
            scheduler = self.pipe.scheduler
            self.refiner.scheduler = scheduler
            image = self.refiner(prompt=item.prompt,
                                negative_prompt=None,
                                num_inference_steps=20,
                                denoising_start=0.8,
                                guidance_scale=7.5,
                                image=image[None, :]).images[0]

            # Convert PIL Image to base64 string
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image = base64.b64encode(buffered.getvalue())

            print("Time Taken:", time.time() - start)

            return {
                "png": image.decode("utf-8"),
            }
        except Exception as e:
            print(e)
            return ""
