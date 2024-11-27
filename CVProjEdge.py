import numpy as np
import cv2
import skimage
from tabulate import tabulate
import scipy.fft
import matplotlib.pyplot as plt

def wavelengthFinder(name, layerName):

    photoName = name
    layer = layerName

    photo = np.array(skimage.io.imread(f'./Results/{photoName}'))
    photo = np.mean(photo, axis=2)

    mask = np.array(skimage.io.imread(f'./Pictures/{layer}'))
    mask = np.mean(mask, axis=2)

    # print(photo.shape)
    # print(mask.shape)

    # skimage.io.imshow(photo, cmap='gray')
    # skimage.io.show()

    # crop at x = 119 to x = 997

    photo_crop = photo[:, 119:997]
    mask_crop = mask[:, 119:997]

    # photo_crop = photo[:, 130:983]
    # photo_crop = photo_crop[14:879, :]

    # mask_crop = mask[:, 130:983]
    # mask_crop = mask_crop[14:879, :]

    # skimage.io.imshow(mask_crop, cmap='gray')
    # skimage.io.show()

    # thr1 = skimage.filters.threshold_otsu(photo)
    # thr2 = skimage.filters.threshold_otsu(mask_crop)

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


    # Now i filter the real photo to ignore the printer nozzle
    # x = 370, y = 92 to x = 556, y = 186
    for row in range(92, 186):
        for col in range(369, 570):
            binaryimg1[row][col] = 0.0

    skimage.io.imshow(binaryimg1, cmap='gray')
    skimage.io.show()
    skimage.io.imshow(binaryimg2, cmap='gray')
    skimage.io.show()

    # get percent error
    # https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python

    # for testing, I black out one of the squares on the original photo

    # for row in range(386, 483):
    #     for col in range(386, 500):
    #         binaryimg1[row][col] = 0.0

    res = cv2.absdiff(binaryimg1, binaryimg2)
    #--- convert the result to integer type ---
    res = res.astype(np.uint8)
    #--- find percentage difference based on the number of pixels that are not zero ---
    ideal_nonzero = np.count_nonzero(binaryimg2)
    percentage = ((ideal_nonzero - (ideal_nonzero - np.count_nonzero(res))) * 100)/ ideal_nonzero
    # print(f"{name}: percent error is: {percentage}%")

    # skimage.io.imshow(res, cmap='gray')
    # skimage.io.show()

    # Now get only the ideal regions on the original photo to take fourier transform of
    masked_photo = np.zeros_like(photo_crop)
    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if binaryimg2[row][col] == 255.0:
                masked_photo[row][col] = photo_crop[row][col]

    # skimage.io.imshow(masked_photo, cmap='gray')
    # skimage.io.show()

    # take fourier transform of this photo

    fourier_image = scipy.fft.fftshift(scipy.fft.fft2(masked_photo))
    #print(np.asarray(fourier_image).shape)
    plt.imshow(20*np.log(np.abs(fourier_image)), cmap='gray')
    plt.title(f"Fourier Transform of {photoName}")
    plt.show()

    # Define area of diagonal frequency locations using an elliptical mask
    mask = np.zeros_like(fourier_image, dtype=np.uint8)

    # define center as center of quadrant II
    center = (int(mask.shape[1]/4), int(mask.shape[0]/4))
    # define axes as length of major axis and minor axis
    axes = (250, 50)
    # angle is rotation clockwise in degrees
    angle = 45

    cv2.ellipse(mask, center, axes, angle, 0, 360, (255, 0, 0), -1)

    result = cv2.bitwise_and(20*np.log(np.abs(fourier_image)), 20*np.log(np.abs(fourier_image)), mask=mask)

    # skimage.io.imshow(mask, cmap='gray')
    # skimage.io.show()

    plt.imshow(result, cmap='gray')
    plt.title(f"Masked Fourier Transform of {photoName}")

    plt.show()

    # Now find location of the maximum in the masked region
    max_index = np.unravel_index(result.argmax(), result.shape)
    #print(f"{name}: Fourier maximum is at: ({max_index[1]}, {max_index[0]}) and is value {result[max_index[0]][max_index[1]]}")
    maxImage = result[max_index[0]][max_index[1]]

    wavelength = 1 / np.sqrt(pow(max_index[0], 2) + pow(max_index[1], 2))
    print(f"{name}: Wavelength for is: {wavelength}")

    fourier_image = scipy.fft.fftshift(scipy.fft.fft2(binaryimg2))
    mask = np.zeros_like(fourier_image, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, (255, 0, 0), -1)
    result = cv2.bitwise_and(20*np.log(np.abs(fourier_image)), 20*np.log(np.abs(fourier_image)), mask=mask)
    max_index = np.unravel_index(result.argmax(), result.shape)
    maxIdeal = result[max_index[0]][max_index[1]]

    #print(f"{name}: Difference = {abs(maxIdeal-maxImage)}")

    # if maxImage > maxIdeal+10:
    #     print("Too high!")
    # elif maxImage < maxIdeal-10:
    #     print("Too low!")
    # else:
    #     print("probably perfect")


wavelengthFinder("functionallyPerfect.png","rectangle.png")
wavelengthFinder("tooHigh.png","rectangle.png")
wavelengthFinder("tooSquished.png","rectangle.png")

wavelengthFinder("perfectBoat.png","boatLayer.png")
wavelengthFinder("highBoat.png","boatLayer.png")
wavelengthFinder("squishedBoat.png","boatLayer.png")