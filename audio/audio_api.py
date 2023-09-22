import sys
import time
import torch
from pydantic import BaseModel
import base64

torch.backends.cuda.matmul.allow_tf32 = True

import modal
from modal import web_endpoint
from ..common import stub

cache_path = "/vol/cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
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
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(cache_path+"/processor")
        self.model = MusicgenForConditionalGeneration.from_pretrained(cache_path+"/model")

    @web_endpoint(method="POST")
    def api(self, item: Item):
        import scipy
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
