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
python -m modal serve modal_shap_e.py
# Change line 84 to match the endpoint URL from modal serve
python modal_shap_e.py --prompt "a dog"
```

## Run with Docker:
```
docker build -t shap-e .
docker run -it --gpus all shap-e python3 app.py --input 'a dog'
```

## Run the Code Locally:
```
python app.py --input 'a dog'
```
