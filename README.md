# tracking_cam
a collaborative effort to bring OpenCV to church

## Heads up...
- this project requires a video feed... for example, a webcam.
- on Mac, switch all "pip" commands to "pip3"

## Dependencies
- Python 3.11.5 / Git
- OpenCV + numpy
- pynput
- Pytorch + Pillow
- Flask

### Dependency CMDs
- git clone https://github.com/jacnok/tracking_cam.git
- pip install opencv-contrib-python==4.8.1.78
- pip install pynput==1.7.6
- pip install facenet-pytorch==2.6.0
- pip install flask==3.0.3

### One-liner for Dependencies
- pip install opencv-contrib-python==4.8.1.78 pynput==1.7.6 facenet-pytorch==2.6.0 flask==3.0.3

#### Semi-Optional Dependencies
- pip install numpy==1.26.1 
    - *(should be installed with the opencv Python package)*
- pip install pillow==10.2.0 
    - *(should be installed with the facenet-pytorch Python package)*