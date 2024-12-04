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
        epochs=75,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        optimizer="auto",    # using better optimizer
        batch=-1,            # auto optimize batch size for GPU
        name="NewerYOLO",    # new name
        lr0=5e-3,            # setting optimalstarting learning rate
        lrf=0.01,            # setting better final learning rate ratio
        weight_decay=0.0005, # default weight decay 
        single_cls=True,     # we only want binary classification and focus on presence of spaghetti
        warmup_epochs=3,     # warmup the testing to smoothen training process
        deterministic=False  # introduce some randomness
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

    # Perform object detection on an image
    results = model("C:/Users/maxwe/Documents/datasets/yolo_data/test/images/1e0889f0d3cb71fe1a1dc15a23bcda8cf78d4241_jpg.rf.3e3ad2892f0df56486caa4ba5a0503f0.jpg")
    results[0].show()

    
if __name__ == '__main__':
    run()