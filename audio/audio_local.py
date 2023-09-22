import base64
import argparse
import requests

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