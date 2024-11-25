from cameraGrabber import currentCamera
from layerGrabber import currentLayer
from skewCorrection import skewCorrection

def main():
    currentCamera()
    currentLayer()
    skewCorrection()

if __name__ == '__main__':
    main()