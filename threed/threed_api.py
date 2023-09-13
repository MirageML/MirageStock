import os
import time
import torch
import requests
import argparse
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
import base64
import typer
import warnings
warnings.filterwarnings("ignore")

from open_source.shap_e.app import generate_3D
from shap_e.models.download import load_model

from modal import Image, Secret, Stub, web_endpoint, create_package_mounts

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    xm = load_model('transmitter', device=device)
    image_model = load_model('image300M', device=device)
    text_model = load_model('text300M', device=device)

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("open_source/shap_e/requirements.txt")
    .run_function(download_models)
)
stub = Stub("stock")

class Item(BaseModel):
    prompt: str = None

@stub.cls(gpu="A10G", image=image, timeout=180)
class ThreeD:
    def __enter__(self):
        self.xm = xm = load_model('transmitter', device=device)
        self.image_model = image_model = load_model('image300M', device=device)
        self.text_model = text_model = load_model('text300M', device=device)

    @web_endpoint(method="POST")
    def api(self, item: Item):
        try:
            start = time.time()
            init_image = False
            if item.prompt.startswith("data:image"):
                item.prompt = item.prompt.replace("data:image/png;base64,", "")
                item.prompt = item.prompt.replace("data:image/jpeg;base64,", "")
                item.prompt = item.prompt.replace("data:image/jpg;base64,", "")
                item.prompt = item.prompt.replace("data:image/gif;base64,", "")
                item.prompt = item.prompt.replace("data:image/bmp;base64,", "")
                item.prompt = item.prompt.replace("data:image/tiff;base64,", "")
                item.prompt = item.prompt.replace("data:image/webp;base64,", "")
                item.prompt = item.prompt.replace("data:image/avif;base64,", "")
                item.prompt = item.prompt.replace("data:image/heif;base64,", "")
                item.prompt = item.prompt.replace("data:image/heic;base64,", "")
                item.prompt = item.prompt.replace("data:image/jxl;base64,", "")

                item.prompt = Image.open(BytesIO(base64.b64decode(item.image))).convert('RGB')
                glb_path = generate_3D(item.prompt, self.image_model, self.xm)
                init_image = True
            else:
                glb_path = generate_3D(item.prompt, self.text_model, self.xm)

            print("Time Taken:", time.time() - start)

            return {
                "glb": str(base64.b64encode(open(glb_path, 'rb').read()).decode('utf-8')),
                "init_image": init_image
            }
        except Exception as e:
            print(e)
            return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="dog")
    args = parser.parse_args()
    data = {"prompt": args.prompt}

    # Change this endpoint to match your own
    response = requests.post("https://mirageml--stock-threed-api-amankishore-dev.modal.run", json=data)
    response = response.json()

    fh = open("mesh.glb", "wb")
    fh.write(base64.b64decode(response["glb"]))
    fh.close()
