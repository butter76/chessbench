## Installing



```
git clone https://github.com/butter76/chessbench.git searchless_chess
python3 -m venv ~/python_env
python3 -m pip install -r requirements.txt
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
```

## Building the engine
```
sudo apt install cmake build-essential libtbb-dev g++-12 libstdc++-12-dev libc++-18-dev libc++abi-18-dev libunwind-18-dev libxxhash-dev
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 -DCMAKE_PREFIX_PATH="/home/shadeform/TensorRT-10.13.2.6;/home/shadeform/TensorRT-10.13.2.6/targets/x86_64-linux-gnu"
cmake --build ./build --target chess_sample -j
```