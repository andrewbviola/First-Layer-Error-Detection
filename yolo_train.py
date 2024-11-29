from ultralytics import YOLO
import torch

def run():

    print(torch.cuda.is_available())
    torch.multiprocessing.freeze_support()

    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="C:/Users/maxwe/Documents/datasets/yolo_data/data.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("spaghetti_5_jpg.rf.da25b28acd7d7b4dc11bcac15899dc85.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

if __name__ == '__main__':
    run()