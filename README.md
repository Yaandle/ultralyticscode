How to run predict.py:
- [git clone](https://github.com/Yaandle/ultralyticscode.git)
  
- Update line 9 with model, and line 51 with video path
 
- *If NOT using GPU, remove 'device' in line 22: 'results = model(frame, conf=0.1, device=0)'
  *If using GPU, ensure Torch is installed with CUDA&CuDNN
 
- python run predict.py
