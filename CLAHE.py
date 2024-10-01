import cv2
import os

def enhance_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def load_images(folder):
    images = []
    for f in os.listdir(folder):
        if not f.endswith("_mask.tif"):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            images.append(enhance_image(img))
    return images
