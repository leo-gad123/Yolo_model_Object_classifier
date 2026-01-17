from ultralytics import YOLO

model=YOLO("yolov8n.pt")

results=model("fruit.jpg",show=True,save=True)
