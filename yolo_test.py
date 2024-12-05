from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

def run():
    print(torch.cuda.is_available())
    torch.multiprocessing.freeze_support()

    model = YOLO("last.pt")

    value = 1 # set to 1 for personal dataset, set to 0 for test dataset

    # testing on the test dataset, images not used before in original training validation
    if value == 0:
        results = model.predict("C:/Users/maxwe/Documents/datasets/yolo_data/test/images",
                            imgsz=640,                          # testing image size
                            device=0,                           # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
                            name="Test Dataset Prediction",     # names the test
                            batch=1)                            # setting batchsize = 1
    
    # testing on our pictures of the printbed
    elif value == 1:
        
        results = model.predict("./Results",
                            imgsz=640,                          # testing image size
                            device=0,                           # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
                            name="Test Dataset Prediction",     # names the test
                            batch=1)                            # setting batchsize = 1
    
    # calculate detection accuracy from the results
    count = 0   

    for i in range(len(results)):
        if value == 0:
            if len(results[i]) == 0: # if value is zero, then we are using the online dataset
                count += 1

                fig, axes = plt.subplots(1, 2)
                results[i].save("image.png")
                img1 = np.array(Image.open("image.png"))
                axes[0].imshow(img1)
                axes[0].set_title("Mislabelled Photo")
                axes[0].axis('off')

                img2 = np.array(Image.open(results[i].path))

                if value == 0:
                    # print(i)
                    filepath = os.path.join("C:/Users/maxwe/Documents/datasets/yolo_data/test/labels", 
                                        os.listdir("C:/Users/maxwe/Documents/datasets/yolo_data/test/labels")[i])

                    # https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/#
                    with open(filepath, "r") as file:
                        lines = file.readlines()
                        for line in lines:
                            # https://stackoverflow.com/questions/66005018/getting-the-second-word-in-a-python-3-string
                            x = int(float(line.split(" ")[1])*640)
                            y = int(float(line.split(" ")[2])*640)
                            # print(x)
                            # print(y)
                            width = int(float(line.split(" ")[3])*640)
                            height = int(float(line.split(" ")[4])*640)

                            rect = patches.Rectangle(((x-0.5*width), (y+0.5*height)), width, -height, linewidth=2, edgecolor='b', facecolor='none')
                            axes[1].add_patch(rect)


                axes[1].imshow(img2)
                axes[1].set_title("Original Photo")
                axes[1].axis('off')

                plt.show()

        elif value == 1:
            if len(results[i]) > 0:
                count += 1
                fig, axes = plt.subplots(1, 2)
                results[i].save("image.png")
                img1 = np.array(Image.open("image.png"))
                axes[0].imshow(img1)
                axes[0].set_title("Mislabelled Photo")
                axes[0].axis('off')

                img2 = np.array(Image.open(results[i].path))

                axes[1].imshow(img2)
                axes[1].set_title("Original Photo")
                axes[1].axis('off')

                plt.show()

    
    percent_error = count / len(results)
    accuracy = (1 - percent_error) * 100
    print(accuracy)
    print(count)

# Make a function to detect spaghetti in an image
def checkSpaghetti(img_path):

    # initialize the model
    model = YOLO("./last.pt")

    # create error flag
    error = False

    # parse through all the images in the path and check for spaghetti
    results = model.predict(img_path,
                            imgsz=640,                          # testing image size
                            device='cpu',                           # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
                            name="Test Dataset Prediction",     # names the test
                            batch=1)                            # setting batchsize = 1

    # look through all the results
    # if any of the results has detecte spaghetti, save the image with spaghetti and display it. Also send an error signal 
    for i in range(len(results)):
        if len(results[i]) > 0:
            error = True
            fig, axes = plt.subplots(1, 2)
            results[i].save("error.png")
            img1 = np.array(Image.open("error.png"))
            axes[0].imshow(img1)
            axes[0].set_title("Mislabelled Photo")
            axes[0].axis('off')

            img2 = np.array(Image.open(results[i].path))

            axes[1].imshow(img2)
            axes[1].set_title("Original Photo")
            axes[1].axis('off')

            plt.show()

    if error == False:
        print("No spaghetti dectected. Continue printing")
    elif error == True:
        print("Spaghetti detected! Stop the print!")

    return error

if __name__ == '__main__':
    run()