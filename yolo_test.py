from tabnanny import check

from ultralytics import YOLO
import time
import requests

# model = YOLO("best.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="./Results/spaghetti.jpg", show=False)
# # results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments
# for result in results:
#     for box in result.boxes:  # Iterate through all detected boxes
#         conf = box.conf[0]  # Confidence score
#         print(f"Confidence: {conf}")


def checkSpaghetti(img,pathModel):
    model = YOLO(pathModel)
    results = model.predict(img)
    for result in results:
        for box in result.boxes:  # Iterate through all detected boxes
            if box.conf[0] > .75:
                return True
    return False

checkSpaghetti("./Results/spaghetti.jpg","best.pt")