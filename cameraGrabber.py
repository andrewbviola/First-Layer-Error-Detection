import cv2
import numpy as np
import requests

def grabImage(url):
    response = requests.get(url)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        return None

def currentCamera(name = "photo.jpg",URL="http://192.168.86.78:8080/photo.jpg"):
    img = grabImage(URL)
    if img is not None:
        cv2.imwrite(f"./Pictures/{name}", img)