import sys
import time
import torch
from io import BytesIO
from pydantic import BaseModel
import base64

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


import modal
from modal import web_endpoint
from ..common import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    # from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
    prior.save_pretrained(cache_path+"/prior", safe_serialization=True)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to(device)
    decoder.save_pretrained(cache_path+"/decoder", safe_serialization=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git"
    )
    .pip_install(
        "transformers",
        "git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887",
        "accelerate",
        "safetensors",
        "xformers",
    )
    .run_function(download_models)
)

class Item(BaseModel):
    prompt: str = None

# Code Adapted from https://github.com/basetenlabs/truss-examples/blob/main/stable-diffusion-xl-1.0
@stub.cls(gpu="A100", image=image, timeout=180)
class Image:
    def __enter__(self):
        # from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler
        from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
        self.prior = StableCascadePriorPipeline.from_pretrained(cache_path+"/prior", torch_dtype=torch.bfloat16).to(device)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(cache_path+"/decoder",  torch_dtype=torch.float16).to(device)

    @web_endpoint(method="POST")
    def api(self, item: Item):
        try:
            start = time.time()
            prior_output = self.prior(
                prompt=item.prompt,
                height=1024,
                width=1024,
                negative_prompt="",
                guidance_scale=4.0,
                num_images_per_prompt=1,
                num_inference_steps=20
            )
            decoder_output = self.decoder(
                image_embeddings=prior_output.image_embeddings.half(),
                prompt=item.prompt,
                negative_prompt="",
                guidance_scale=0.0,
                output_type="pil",
                num_inference_steps=10
            ).images
            image = decoder_output[0]

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
