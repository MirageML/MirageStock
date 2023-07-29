#Run the Code with Modal:
##First install Modal locally (modal.com):
```
pip install modal-client
modal token new
```
##Then run the code:
```
pip install -r requirements.txt
python3 -m modal serve modal_shap_e.py
python3 modal_shap_e.py --prompt "a dog"
```

Run with Docker:
```
docker build -t shap-e .
docker run -it --gpus all shap-e python3 app.py --input 'a dog'
```
```

Run the Code Locally:
```
python3 app.py --input 'a dog'
```