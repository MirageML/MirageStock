# Run the Code with [Modal](https://modal.com):
## First install Modal locally:
```
pip install modal-client
modal token new
```
## Then run the code:
### The prompt can be text or a base64 string version of an image
```
python -m modal serve MirageStock        # Executed at the Top-Level 
python threed_local.py --prompt "a dog"  # Change line 12 to match the endpoint URL from modal serve
```
