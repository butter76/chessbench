## Installing



```
git clone https://github.com/butter76/chessbench.git searchless_chess
python3 -m venv ~/python_env
python3 -m pip install -r requirements.txt
python3 -m pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu122
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install flash-attn
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
```
