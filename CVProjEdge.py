import math
import pandas as pd
import numpy as np
import cv2
import skimage
from tabulate import tabulate
import matplotlib.pyplot as plt

photo = np.array(skimage.io.imread('./cv_photos/real.png'))
photo = np.mean(photo, axis=2)

mask = np.array(skimage.io.imread('./cv_photos/mask.png'))
mask = np.mean(mask, axis=2)

# print(photo.shape)
# print(mask.shape)

skimage.io.imshow(photo, cmap='gray')
skimage.io.show()

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
        if photo_crop[row][col] < 60:
            binaryimg1[row][col] = 255
        else:
            binaryimg1[row][col] = 0
        if mask_crop[row][col] > 120:    
            binaryimg2[row][col] = 255
        else:
            binaryimg2[row][col] = 0


# Now i filter the real photo to ignore the printer nozzle
# x = 370, y = 92 to x = 556, y = 186
for row in range(92, 186):
    for col in range(369, 570):
        binaryimg1[row][col] = 0.0

# skimage.io.imshow(binaryimg1, cmap='gray')
# skimage.io.show()
# skimage.io.imshow(binaryimg2, cmap='gray')
# skimage.io.show()

# get percent error
# https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python

# for testing i black out one of the squares on the original photo

for row in range(386, 483):
    for col in range(386, 500):
        binaryimg1[row][col] = 0.0

res = cv2.absdiff(binaryimg1, binaryimg2)
#--- convert the result to integer type ---
res = res.astype(np.uint8)
#--- find percentage difference based on the number of pixels that are not zero ---
ideal_nonzero = np.count_nonzero(binaryimg2)
percentage = ((ideal_nonzero - (ideal_nonzero - np.count_nonzero(res))) * 100)/ ideal_nonzero
print(f"percent error is: {percentage}%")

skimage.io.imshow(res, cmap='gray')
skimage.io.show()






