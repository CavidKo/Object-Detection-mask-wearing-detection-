from ultralytics import YOLO


model = YOLO('maskDetector.pt')

results = model(source=r'\test\image\masks-tokyo.jpg', show=True, conf=0.3, save=True) # source = <link to file or 0 for the camera>