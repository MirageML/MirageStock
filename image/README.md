# Run the Code with [Modal](https://modal.com):
## First install Modal locally:
```
pip install modal-client
modal token new
```
## Then run the code:
### The prompt can be text or a base64 string version of an image
```
pip install -r requirements.txt
python -m modal serve image_api.py
# Change line 84 to match the endpoint URL from modal serve
python image_api.py --prompt "a dog"
```

