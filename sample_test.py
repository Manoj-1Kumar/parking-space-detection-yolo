from ultralytics import YOLO

model = YOLO(r"D:\yolo1\runs\detect\train\weights\best.pt")

model(
    source=r"D:\yolo1\sample_test_img",
    save=True,
    conf=0.4
)
