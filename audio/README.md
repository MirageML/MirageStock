# Run the Code with [Modal](https://modal.com):
## First install Modal locally:
```
pip install modal-client
modal token new
```
## Then run the code:
```
python -m modal serve MirageStock                # Executed at the Top-Level 
python audio_api.py --prompt "a classical song"  # Change line 12 to match the endpoint URL from modal serve
```

