import numpy as np
import cv2
import skimage

def missingObject(imagePath, layerPath):
    photoName = imagePath
    layer = layerPath

    photo = np.array(skimage.io.imread(f'{photoName}'))
    photo = np.mean(photo, axis=2)

    mask = np.array(skimage.io.imread(f'{layer}'))
    mask = np.mean(mask, axis=2)

    photo_crop = photo[20:870, 125:997]
    mask_crop = mask[20:870, 125:997]

    # photo_crop = photo[:, 130:983]
    # photo_crop = photo_crop[14:879, :]

    # mask_crop = mask[:, 130:983]
    # mask_crop = mask_crop[14:879, :]


    # thr1 = skimage.filters.threshold_otsu(photo)
    # thr2 = skimage.filters.threshold_otsu(mask_crop)

    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if mask_crop[row][col] < 100:
                photo_crop[row][col] = 0
                mask_crop[row][col] = 0
            else:
                mask_crop[row][col] = 255

    skimage.io.imshow(photo_crop, cmap='gray')
    skimage.io.show()

    threshold = np.max(photo_crop)*.10

    for row in range(photo_crop.shape[0]):
        for col in range(photo_crop.shape[1]):
            if photo_crop[row][col] > threshold:
                photo_crop[row][col] = 255
            else:
                if photo_crop[row][col] > 1:
                    photo_crop[row][col] = 0



    # Now i filter the real photo to ignore the printer nozzle
    # x = 370, y = 92 to x = 556, y = 186

    skimage.io.imshow(photo_crop, cmap='gray')
    skimage.io.show()
    skimage.io.imshow(mask_crop, cmap='gray')
    skimage.io.show()

    # get percent error
    # https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python

    # for testing, I black out one of the squares on the original photo

    # for row in range(386, 483):
    #     for col in range(386, 500):
    #         binaryimg1[row][col] = 0.0

    res = cv2.absdiff(photo_crop, mask_crop)
    # --- convert the result to integer type ---
    res = res.astype(np.uint8)
    # --- find percentage difference based on the number of pixels that are not zero ---
    ideal_nonzero = np.count_nonzero(mask_crop)
    percentage = ((ideal_nonzero - (ideal_nonzero - np.count_nonzero(res))) * 100) / ideal_nonzero
    #print(f"percent error is: {percentage}%")
    if percentage > 25 or percentage == 0.0:
        print(f"Percent error of {round(percentage,2)}, likely missing objects")
    else:
        print(f"Percent error of {round(percentage,2)}, continue as normal")



if __name__ == '__main__':
    missingObject('missingFox.png', 'missingFoxTest.png')
    missingObject("missingPS5.png","ps5.png")
    missingObject("missingBenchy.png", "benchy.png")
    missingObject('perfectTest.png', 'test.png')
    missingObject('workingPS5.png',"ps5.png")
    missingObject("workingBenchy.png","benchy.png")