import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def edgeDetection(camera,layer,upper,lower):

    photo = np.array(skimage.io.imread(camera))
    photo = np.mean(photo, axis=2)
    photo = photo.astype(np.uint8)

    for col in range(410,800):
        for row in range(0,160):
            photo[row,col] = 120

    mask = np.array(skimage.io.imread(layer))
    mask = np.mean(mask, axis=2)
    mask = mask.astype(np.uint8)

    photo_crop = photo[:, 119:997]
    mask_crop = mask[:, 119:997]

    binaryimg1 = np.zeros_like(photo_crop)
    binaryimg2 = np.zeros_like(mask_crop)

    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if photo_crop[row][col] < 55:
                binaryimg1[row][col] = 255
            else:
                binaryimg1[row][col] = 0
            if mask_crop[row][col] > 85:
                binaryimg2[row][col] = 255
            else:
                binaryimg2[row][col] = 0

    masked_photo = np.zeros_like(photo_crop)
    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if binaryimg2[row][col] == 255.0:
                masked_photo[row][col] = photo_crop[row][col]

    masked_layer = np.zeros_like(photo_crop)
    for row in range(masked_layer.shape[0]):
        for col in range(masked_layer.shape[1]):
            if binaryimg2[row][col] == 255.0:
                masked_layer[row][col] = mask[row][col]

    edgeCam = cv2.Canny(binaryimg1,lower,upper)
    edgeLayer = cv2.Canny(binaryimg2,lower,upper)

    plt.imshow(edgeCam)
    plt.title("Camera")
    plt.show()
    plt.imshow(edgeLayer)
    plt.title("Perfect Layer")
    plt.show()

    squared_diff = (masked_photo.astype(np.float32) - masked_layer.astype(np.float32)) ** 2
    mse = np.mean(squared_diff)
    ssim_value, ssim_map = ssim(masked_photo, masked_layer, full=True)
    print(f"{camera}: SSIM = {ssim_value}")
    print(f"{camera}: mse = {mse}")
    print("\n")

edgeDetection("./Results/perfectBoat.png","./Pictures/boatLayer.png",40,150)
edgeDetection("./Results/highBoat.png","./Pictures/boatLayer.png",40,150)
edgeDetection("./Results/squishedBoat.png","./Pictures/boatLayer.png",40,150)

edgeDetection("./Results/nearPerfect.png","./Pictures/rectangle.png",40,150)
edgeDetection("./Results/tooHigh.png","./Pictures/rectangle.png",40,150)
edgeDetection("./Results/tooSquished.png","./Pictures/rectangle.png",40,150)