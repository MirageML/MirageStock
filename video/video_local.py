import base64
import argparse
import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a dog running")
    args = parser.parse_args()
    data = {"prompt": args.prompt}

    # Change this endpoint to match your own
    response = requests.post("https://mirageml--stock-video-api-amankishore-dev.modal.run", json=data)
    responseData = response.json()

    # Save mp4 to file
    fh = open("video.mp4", "wb")
    fh.write(base64.b64decode(responseData["mp4"]))
    fh.close()