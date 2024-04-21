How to run Ultralytics Python code:
- [git clone](https://github.com/Yaandle/ultralyticscode.git)

- Use Image or Video inference.
  
- Update lines with model, and with video path
 
- *If NOT using GPU, remove 'device' in line: 'results = model(frame, conf=0.1, device=0)'
- *If using GPU, ensure Torch is installed with CUDA&CuDNN
 
- python run _app.py_
