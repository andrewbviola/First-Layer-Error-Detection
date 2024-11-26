from cameraGrabber import currentCamera
from layerGrabber import currentLayer
from skewCorrection import skewCorrection

def main():
    currentCamera()
    print("Camera grabbed")
    currentLayer()
    print("Layer grabbed")
    skewCorrection()

if __name__ == '__main__':
    main()