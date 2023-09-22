import base64
import argparse
import requests

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
