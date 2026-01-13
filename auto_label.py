import xml.etree.ElementTree as ET
import os

XML_FILE = "my_dataset/annotations.xml"
OUT_DIR = "parking_yolo/labels/train"

os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP = {
    "free_parking_space": 0,
    "not_free_parking_space": 1,
    "partially_free_parking_space": 2
}

tree = ET.parse(XML_FILE)
root = tree.getroot()

total_boxes = 0
total_images = 0

for image in root.iter("image"):
    total_images += 1
    img_name = os.path.basename(image.get("name"))
    img_w = float(image.get("width"))
    img_h = float(image.get("height"))

    label_file = os.path.join(
        OUT_DIR, img_name.replace(".png", ".txt")
    )

    lines = []

    for poly in image.iter("polygon"):
        label = poly.get("label")
        pts = poly.get("points")

        if label not in CLASS_MAP or pts is None:
            continue

        xs, ys = [], []
        for p in pts.split(";"):
            x, y = map(float, p.split(","))
            xs.append(x)
            ys.append(y)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if xmax <= xmin or ymax <= ymin:
            continue

        xc = ((xmin + xmax) / 2) / img_w
        yc = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f"{CLASS_MAP[label]} {xc} {yc} {bw} {bh}")
        total_boxes += 1

    with open(label_file, "w") as f:
        f.write("\n".join(lines))

print("Images processed:", total_images)
print("Total boxes written:", total_boxes)