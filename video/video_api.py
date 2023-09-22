import os
import sys
import time
import torch
from io import BytesIO
from pydantic import BaseModel
import base64
import typer
from typing import Any

import modal
from modal import web_endpoint
from ..common import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_models():
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained(cache_path+"/pipe", safe_serialization=True)

    pipexl = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
    pipexl.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipexl.save_pretrained(cache_path+"/pipexl", safe_serialization=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg"
    )
    .pip_install(
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "xformers",
        "opencv-python",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "einops",
        "omegaconf",
        "decord"
    )
    .run_function(download_models)
)

class Item(BaseModel):
    prompt: str = None

@stub.cls(gpu="A100", image=image, timeout=180)
class Video:
    def __enter__(self):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        self.pipe = DiffusionPipeline.from_pretrained(cache_path+"/pipe", torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

        self.pipexl = DiffusionPipeline.from_pretrained(cache_path+"/pipexl", torch_dtype=torch.float16)
        self.pipexl.scheduler = DPMSolverMultistepScheduler.from_config(self.pipexl.scheduler.config)
        self.pipexl.enable_model_cpu_offload()
        self.pipexl.enable_vae_slicing()


    @web_endpoint(method="POST")
    def api(self, item: Item):
        from PIL import Image
        from diffusers.utils import export_to_video
        try:
            start = time.time()

            video_frames = self.pipe(item.prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
            video_path = export_to_video(video_frames)

            video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

            video_frames = self.pipexl(item.prompt, video=video, strength=0.6).frames
            video_path = export_to_video(video_frames)

            # No idea why this works found it here: https://github.com/camenduru/text-to-video-synthesis-colab/blob/main/potat1_text_to_video_diffusers.ipynb
            os.system(f"ffmpeg -y -i {video_path} -c:v libx264 -c:a aac -strict -2 /tmp/fixed.mp4 >/dev/null 2>&1")

            print("Time Taken:", time.time() - start)

            return {
                "mp4": str(base64.b64encode(open("/tmp/fixed.mp4", 'rb').read()).decode('utf-8')),
            }
        except Exception as e:
            print(e)
            return str(e)
