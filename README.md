## Installing



```
git clone https://github.com/butter76/chessbench.git searchless_chess
python3 -m venv ~/python_env
python3 -m pip install -r requirements.txt
python3 -m pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu122
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install flash-attn
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
export LLVM_BUILD_DIR=$HOME/llvm-project/build
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include   LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib   LLVM_SYSPATH=$LLVM_BUILD_DIR   python3 -m pip install -e python
```