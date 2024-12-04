from ultralytics import YOLO
import torch

def run():
    print(torch.cuda.is_available())
    torch.multiprocessing.freeze_support()

    model = YOLO("./runs/detect/NewYOLO/weights/last.pt")

    # for testing on individual images
    # results = model.predict(source="./cv_photos/real.png", show=True)
    # results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments
    # results[0].show()

    # testing on the test dataset, was not used before in original training validation
    results = model.predict("C:/Users/maxwe/Documents/datasets/yolo_data/test/images",
                        imgsz=640,                          # training image size
                        device=0,                           # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
                        name="Test Dataset Prediction",     # names the test
                        batch=1)                           # setting batchsize = 1
    
    # calculate detection accuracy from the results
    count = 0

    for i in range(len(results)):
        if len(results[i]) == 0:
            count += 1
    
    percent_error = count / len(results)
    accuracy = (1 - percent_error) * 100
    print(accuracy)

if __name__ == '__main__':
    run()