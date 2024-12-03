from ultralytics import YOLO
import torch

def run():
    print(torch.cuda.is_available())
    torch.multiprocessing.freeze_support()

    model = YOLO("./runs/detect/train14/weights/best.pt")

    # for testing on individual images
    # results = model.predict(source="./cv_photos/real.png", show=True)
    # results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments
    # results[0].show()

    # Testing on the test dataset, was not used before in original training validation

    metrics = model.val(data="C:/Users/maxwe/Documents/datasets/yolo_data/data.yaml",
                        imgsz=640,                          # training image size
                        device=0,                           # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
                        plots=True,                         # creates a plot of prediction vs ground truth  
                        name="Test Dataset Evaluation",     # names the test
                        split="test")                       # using test dataset

if __name__ == '__main__':
    run()