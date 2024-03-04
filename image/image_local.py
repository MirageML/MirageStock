import base64
import argparse
import requests
from io import BytesIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="dog")
    args = parser.parse_args()
    data = {"prompt": args.prompt}

    # Change this endpoint to match your own
    response = requests.post("https://mirageml--stock-image-api-dev.modal.run", json=data)
    response = response.json()

    # Save image to file
    img_data = base64.b64decode(response["png"])
    from PIL import Image
    img = Image.open(BytesIO(img_data))
    img.save("image.png")