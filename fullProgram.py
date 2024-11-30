from cameraGrabber import currentCamera
from layerGrabber import currentLayer
from skewCorrection import skewCorrection
from gradients import imageGradients
from missingObject import missingObject
from yolo_test import checkSpaghetti
import time
import requests

def main():
    url = "http://enderwire.local/printer/"
    currentCamera()
    print("Camera grabbed")
    currentLayer()
    print("Layer grabbed")
    skewCorrection()
    missingObject("skewCorrection.png", "layer.png")
    # imageGradients("skewCorrection.png","layer.png")
    #
    # requests.post(f"{url}print/resume")
    # print("Waiting for resume")
    # while requests.get(f"{url}objects/query?print_stats").json()["result"]["status"]["print_stats"]["state"] == "paused":
    #     time.sleep(1)
    #
    # while requests.get(f"{url}objects/query?print_stats").json()["result"]["status"]["print_stats"]["state"] != "printing":
    #     print("Waiting for pause")
    #     time.sleep(60)
    #     requests.post(f"{url}print/pause")
    #     currentCamera()
    #     currentLayer()
    #     skewCorrection()
    #     if checkSpaghetti("skewCorrection.png","best.pt"):
    #         print("Spaghetti Found!")
    #     else:
    #         print("Spaghetti Not Found!")
    #         requests.post(f"{url}print/resume")
    #         while requests.get(f"{url}objects/query?print_stats").json()["result"]["status"]["print_stats"]["state"] == "paused":
    #             time.sleep(1)



if __name__ == '__main__':
    main()