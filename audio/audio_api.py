import os
import sys
import time
import torch
import requests
import argparse
from io import BytesIO
from pydantic import BaseModel
import base64
import scipy
import typer
from typing import Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration

torch.backends.cuda.matmul.allow_tf32 = True

import modal
from modal import web_endpoint
sys.path.insert(0, '../..')
from MirageStock import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    processor.save_pretrained(cache_path+"/processor", safe_serialization=True)

    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.save_pretrained(cache_path+"/model", safe_serialization=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers",
        "accelerate",
        "safetensors",
        "xformers",
        "scipy"
    )
    .run_function(download_models)
)

class Item(BaseModel):
    prompt: str = None

@stub.cls(gpu="A10G", image=image, timeout=180)
class Audio:
    def __enter__(self):
        self.processor = AutoProcessor.from_pretrained(cache_path+"/processor")
        self.model = MusicgenForConditionalGeneration.from_pretrained(cache_path+"/model")

    @web_endpoint(method="POST")
    def api(self, item: Item):
        try:
            start = time.time()
            inputs = self.processor(
                text=[item.prompt],
                padding=True,
                return_tensors="pt",
            )

            audio_values = self.model.generate(**inputs, max_new_tokens=256)
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            scipy.io.wavfile.write("/tmp/musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


            # Convert musicgen_out.wav to a base64 string

            print("Time Taken:", time.time() - start)

            return {
                "wav": str(base64.b64encode(open("/tmp/musicgen_out.wav", 'rb').read()).decode('utf-8')),
            }
        except Exception as e:
            print(e)
            return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a classical song")
    args = parser.parse_args()
    data = {"prompt": args.prompt}

    # Change this endpoint to match your own
    response = requests.post("https://mirageml--stock-audio-api-amankishore-dev.modal.run", json=data)
    response = response.json()

    # Save wav base64 to file
    wav_data = base64.b64decode(response["wav"])
    fh = open("music.wav", "wb")
    fh.write(wav_data)
    fh.close()
