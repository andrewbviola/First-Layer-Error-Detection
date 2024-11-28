import cv2
import skimage
import numpy as np

def resizeROA(name,layerName):
    photoName = name
    layer = layerName

    photo = np.array(skimage.io.imread(f'./Results/{photoName}'))
    photo = np.mean(photo, axis=2)

    mask = np.array(skimage.io.imread(f'./Pictures/{layer}'))
    mask = np.mean(mask, axis=2)

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


    for row in range(92, 186):
        for col in range(369, 570):
            binaryimg1[row][col] = 0.0

    height, width = photo.shape[:2]
    x_center, y_center = width // 2, height // 2
    zoom_factor = 2  # How much to zoom (e.g., 2x)

    # Calculate the cropping coordinates
    crop_width = width // (2 * zoom_factor)
    crop_height = height // (2 * zoom_factor)

    x1, y1 = x_center - crop_width, y_center - crop_height
    x2, y2 = x_center + crop_width, y_center + crop_height

    # Crop the image
    cropped = photo[y1:y2, x1:x2]

    # Resize back to original dimensions (optional, for simulated zoom)
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(layerName, zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


resizeROA('perfectBoat.png',"boatLayer.png")