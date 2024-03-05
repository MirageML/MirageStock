import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
import base64
import tempfile
import requests
import warnings
warnings.filterwarnings("ignore")

import modal
from modal import web_endpoint
from ..common import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def download_models():
#     HF_TOKEN = os.getenv("HF_TOKEN")
#     model = TSR.from_pretrained(
#         "stabilityai/TripoSR",
#         config_name="config.yaml",
#         weight_name="model.ckpt",
#         token=HF_TOKEN
#     )
#     model_state_dict = model.state_dict()

#     # Define the directory path where you want to save the model
#     save_dir = 'cache_path+"/tsr"'
#     os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

#     # Define the file path for saving the model
#     save_path = os.path.join(save_dir, 'model.pth')

#     # Save the model state dictionary
#     torch.save(model_state_dict, save_path)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
    )
    .pip_install(
        "omegaconf==2.3.0",
        "Pillow==10.1.0",
        "einops==0.7.0",
        "git+https://github.com/tatsy/torchmcubes.git",
        "transformers==4.35.0",
        "trimesh",
        "rembg",
        "huggingface-hub"
    )
    # .run_function(download_models, timeout=600)
)

class Item(BaseModel):
    prompt: str = None

@stub.cls(keep_warm=3, image=image, timeout=600, mounts=[modal.Mount.from_local_dir("/Users/amankishore/Documents/Git/Mirage-git/MirageStock/threed/tsr", remote_path="/root/MirageStock/threed/tsr")])
class ThreeD:
    def __enter__(self):
        from .tsr.system import TSR
        HF_TOKEN = os.getenv("HF_TOKEN")
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
            token=HF_TOKEN
        )
        # self.model = TSR.from_pretrained(cache_path+"/tsr")
        self.model.renderer.set_chunk_size(131072)
        self.model.to("cpu")

    def preprocess(self, input_image, do_remove_background=True, foreground_ratio=0.85):
        from .tsr.utils import remove_background, resize_foreground
        import rembg

        rembg_session = rembg.new_session()

        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image

        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
        return image

    def generate(self, image):
        from .tsr.utils import to_gradio_3d_orientation
        scene_codes = self.model(image, device=device)
        mesh = self.model.extract_mesh(scene_codes)[0]
        mesh = to_gradio_3d_orientation(mesh)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(mesh_path.name, file_type='glb')
        return mesh_path.name

    @web_endpoint(method="POST")
    def api(self, item: Item):
        # try:
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
            preprocessed = self.preprocess(item.prompt, True, 0.9)

            glb_path = self.generate(preprocessed)
            init_image = True
        else:
            data = {"prompt": item.prompt}

            # Change this endpoint to match your own
            response = requests.post("https://mirageml--stock-image-api.modal.run", json=data)
            response = response.json()

            # Save image to file
            item.prompt = Image.open(BytesIO(base64.b64decode(response["png"]))).convert('RGB')
            preprocessed = self.preprocess(item.prompt, True, 0.9)
            glb_path = self.generate(preprocessed)


        print("Time Taken:", time.time() - start)

        return {
            "glb": str(base64.b64encode(open(glb_path, 'rb').read()).decode('utf-8')),
            "init_image": init_image
        }
        # except Exception as e:
        #     print(e)
        #     return ""
