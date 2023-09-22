import sys
import time
import torch
from io import BytesIO
from pydantic import BaseModel
import base64
import warnings
warnings.filterwarnings("ignore")

import modal
from modal import web_endpoint
from ..common import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    from shap_e.models.download import load_model
    xm = load_model('transmitter', device=device)
    image_model = load_model('image300M', device=device)
    text_model = load_model('text300M', device=device)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "pyyaml",
        "ipywidgets",
        "git+https://github.com/openai/shap-e.git",
        "trimesh",
        "matplotlib"
    )
    .run_function(download_models)
)

class Item(BaseModel):
    prompt: str = None

@stub.cls(gpu="A100", image=image, timeout=180)
class ThreeD:
    def __enter__(self):
        from shap_e.models.download import load_model
        self.xm = xm = load_model('transmitter', device=device)
        self.image_model = image_model = load_model('image300M', device=device)
        self.text_model = text_model = load_model('text300M', device=device)

    @web_endpoint(method="POST")
    def api(self, item: Item):
        from PIL import Image
        from .open_source.shap_e.app import generate_3D
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
