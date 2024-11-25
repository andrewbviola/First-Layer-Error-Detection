import cv2
import numpy as np

def skewCorrection():

    cameraImg = cv2.imread("./Pictures/photo.jpg")
    layerImg = cv2.imread("./Pictures/layer.png")

    cameraPoints = np.array([
        [709,1050],
        [1639,1062],
        [1641,129],
        [731,114]
    ], dtype='float32')

    layerPoints = np.array([
        [122,885],
        [992,885],
        [992,8],
        [122,8]
    ], dtype='float32')

    homography = cv2.getPerspectiveTransform(cameraPoints,layerPoints)

    newImg = cv2.warpPerspective(cameraImg,homography,(layerImg.shape[1],layerImg.shape[0]))

    blend = cv2.addWeighted(layerImg,1,newImg,0.5,0)

    cv2.imshow("Skew Correction",newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f"./Results/skewCorrection.png",newImg)

    cv2.imshow("Overlap",blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

