How to run Ultralytics Python code:
- [git clone](https://github.com/Yaandle/ultralyticscode.git)

- Use either Image, Video or Stream inference, each has their own app.


Code Setup
- Change the model path
- For Image and Video change the source.
- *If NOT using GPU, remove 'device' in line: 'results = model(frame, conf=0.1, device=0)'
- *If using GPU, ensure Torch is installed with CUDA&CuDNN
 
- python run _app.py_
