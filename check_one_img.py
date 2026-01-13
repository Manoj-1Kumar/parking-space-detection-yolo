from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model("parking_yolo/images/train/5.png", save=True)
