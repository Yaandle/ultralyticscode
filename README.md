How to run predict.py:
1. [git clone](https://github.com/Yaandle/ultralyticscode.git)
2. 
3. Update line 9 with model, and line 51 with video path
4. 
5. *If NOT using GPU, remove 'device' in line 22: 'results = model(frame, conf=0.1, device=0)'
3. *If using GPU, ensure Torch is installed with CUDA&CuDNN
4. 
5. python run predict.py
