# Run the Code with [Modal](https://modal.com):
## First install Modal locally:
```
pip install modal-client
modal token new
```
## Then run the code:
```
python -m modal serve MirageStock               # Executed at the Top-Level
python video_local.py --prompt "a dog running"  # Change line 84 to match the endpoint URL from modal serve
```
