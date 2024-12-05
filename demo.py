from cameraGrabber import currentCamera
from layerGrabber import currentLayer
from skewCorrection import skewCorrection
from gradients import imageGradients
from missingObject import missingObject
from yolo_test import checkSpaghetti
import sys
import os

# First take photo of a first layer. This will not work in the function but is provided commented out
# Instead you will be provided with a pretend picture set up for this test as connection to the printer we are using is not available.

# url = "http://enderwire.local/printer/"
# currentCamera()
# print("Camera grabbed")

def demo():
    # By here, a picture from the camera is taken. For this test an image is provided here:
    cameraImgPath = "./Demo/tooHigh.jpg"

    # currentLayer()
    # print("Layer grabbed")

    # By here, a picture from the current layer is taken. For this test an image is provided here:
    layerImgPath = "./Demo/rectangle.png"

    # Now we skew correct the camera image to the layer image with a 2D homography. This is saved to the demo folder as skewCorrection.png
    skewCorrection(cameraImgPath, layerImgPath)

    # After we get a clean image of the build plate, we will check for spaghetti with our pretrained YOLO model. To see how this was trained, see yolo_train.py
    #error = checkSpaghetti("./Demo/skewCorrection.png")

    # if error == True:
    #     print("Halting Print...")
    #     # Send signal to printer to stop via Moonraker’s API
    #     # Halt code execution
    #     sys.exit()
        
    # Next, the two images are checked if any objects are missing
    missingObject("./Demo/skewCorrection.png", "./Demo/rectangle.png")

    # Finally, we check the quality of the first layer printed with the gradient analysis
    imageGradients("./Demo/skewCorrection.png", "./Demo/rectangle.png")


if __name__ == '__main__':
    demo()
