import numpy as np
import cv2
import skimage
from tabulate import tabulate
import scipy.fft
import matplotlib.pyplot as plt

def imageGradients(imageName,layerName):
    photoName = imageName
    layer = layerName

    photo = np.array(skimage.io.imread(f'./Results/{photoName}'))
    photo = np.mean(photo, axis=2)

    mask = np.array(skimage.io.imread(f'./Pictures/{layer}'))
    mask = np.mean(mask, axis=2)

    photo_crop = photo[:, 119:997]
    mask_crop = mask[:, 119:997]
    #
    # skimage.io.imshow(photo_crop, cmap='gray')
    # skimage.io.show()

    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if mask_crop[row][col] < 100:
                photo_crop[row][col] = 0

    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if photo_crop[row][col] > 70:
                photo_crop[row][col] = 0

    gradX, gradY = np.gradient(photo_crop)
    gradients = np.sqrt(gradX ** 2 + gradY ** 2)
    meanGrad = np.mean(gradients)
    varGrad = np.var(gradients)

    # print(f"{imageName} mean = {meanGrad}, var = {varGrad}")
    photo_crop = photo_crop.astype(np.uint8)
    edges = cv2.Canny(photo_crop, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size


    print(f"{imageName[:5]} edge = {round(edge_density,5)}, mean = {round(meanGrad,5)}, var = {round(varGrad,5)}")


    if .001 < edge_density < .0035 and (0.225 < meanGrad < 0.6 or 1 < varGrad < 2):
        print(f"{imageName}: Ok!")
    elif edge_density < .0009 or (.22 < meanGrad < .62 and 2.5 < varGrad < 5.5):
        print(f"{imageName}: Too squished!")
    elif edge_density > 0.0022 and varGrad > 3 and meanGrad > 0.1:
       print(f"{imageName}: Too high!")

    # plt.imshow(gradients)
    # plt.title(imageName)
    # plt.show()
    #
    # skimage.io.imshow(photo_crop, cmap='gray')
    # skimage.io.show()

imageGradients("functionallyPerfect.png", "rectangle.png")
imageGradients("perfectVoron.png", "voron.png")
imageGradients("perfectBoat.png", "boatLayer.png")

imageGradients("tooSquished.png", "rectangle.png")
imageGradients("squishedVoron.png", "voron.png")
imageGradients("squishedBoat.png", "boatLayer.png")

imageGradients("highBoat.png", "boatLayer.png")
imageGradients("tooHigh.png", "rectangle.png")
imageGradients("highVoron.png", "voron.png")
