import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def skewCorrection(img,layer):
    image1 = cv2.imread(f"./Results/{img}", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(f"./Pictures/{layer}", cv2.IMREAD_GRAYSCALE)

    image1 = image1[:870, 125:950]
    image2 = image2[:870, 125:950]

    outImage1 = np.copy(image1)

    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            if image2[row, col] > 85:
                image2[row, col] = 255
                for row2 in range(row-1,row+1):
                    for col2 in range(col-1,col+1):
                        if image1[row2,col2] < 85 or image1[row2,col2] < 255:
                            outImage1[row2,col2] = 255
                        else:
                            outImage1[row2,col2] = 0
            else:
                image2[row, col] = 0
                outImage1[row, col] = 0


    output = ssim(outImage1, image2)

    print(output)

if __name__ == '__main__':
    skewCorrection("functionallyPerfect.png", "rectangle.png")
    skewCorrection("perfectVoron.png", "voron.png")
    skewCorrection("perfectBoat.png", "boatLayer.png")
    skewCorrection("perfectHotend.png", "hotend.png")
    skewCorrection("perfectTest.png", "test.png")
    skewCorrection("functionalExtruder.png", "extruder.png")
    skewCorrection("perfectFox.png", "fox.png")
    skewCorrection("perfectFlower.png", "flower.png")
    skewCorrection("perfectWatch.png", "watch.png")

    skewCorrection("tooSquished.png", "rectangle.png")
    skewCorrection("squishedVoron.png", "voron.png")
    skewCorrection("squishedBoat.png", "boatLayer.png")
    skewCorrection("squishedHotend.png", "hotend.png")
    skewCorrection("squishedTest.png", "test.png")
    skewCorrection("squishedExtruder.png", "extruder.png")
    skewCorrection("squishedFox.png", "fox.png")
    skewCorrection("squishedFlower.png","flower.png")
    skewCorrection("squishedWatch.png", "watch.png")
    skewCorrection("squishedPrusa.png","prusa.png")

    skewCorrection("highBoat.png", "boatLayer.png")
    skewCorrection("tooHigh.png", "rectangle.png")
    skewCorrection("highVoron.png", "voron.png")
    skewCorrection("highHotend.png", "hotend.png")
    skewCorrection("highTest.png", "test.png")
    skewCorrection("highExtruder.png","extruder.png")
    skewCorrection("highFox.png", "fox.png")
    skewCorrection("highFlower.png","flower.png")
    skewCorrection("highWatch.png", "watch.png")
    skewCorrection("highPrusa.png","prusa.png")
