from ultralytics import YOLO
import cv2

model = YOLO(r"D:\yolo1\runs\detect\train\weights\best.pt")

cap = cv2.VideoCapture(r"D:\yolo1\sample_test_vid\1000097099.mp4")  # webcam or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]

    free_count = 0
    total = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        total += 1
        if cls == 0:
            free_count += 1

    
    cv2.putText(
        frame,
        f"Free Spaces: {free_count}/{total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Draw detections
    annotated = results.plot()

    cv2.imshow("Smart Parking Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()