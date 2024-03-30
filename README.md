How to run predict.py:
1. git clone
2. *If NOT using GPU, remove 'device' in line 22: 'results = model(frame, conf=0.1, device=0)'
3. *If using GPU, ensure Torch is installed with CUDA&CuDNN
2. python run predict.py
