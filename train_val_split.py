import os, random, shutil

train_img = "parking_yolo/images/train"
train_lbl = "parking_yolo/labels/train"
val_img = "parking_yolo/images/val"
val_lbl = "parking_yolo/labels/val"

os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = os.listdir(train_img)
val_images = random.sample(images, 6)

for img in val_images:
    shutil.move(f"{train_img}/{img}", f"{val_img}/{img}")
    shutil.move(f"{train_lbl}/{img.replace('.png','.txt')}",
                f"{val_lbl}/{img.replace('.png','.txt')}")

