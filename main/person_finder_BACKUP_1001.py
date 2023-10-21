import cv2
import numpy as np

# imported python modules as per YT video: 
# https://www.youtube.com/watch?v=rKcwcARdg9M

# followed this tutorial: https://www.datacamp.com/tutorial/face-detection-python-opencv

cap = cv2.VideoCapture(0)                                               # init webcam

face_classifier = cv2.CascadeClassifier(                                # init cascadeclassifier: this is the model we use to target what body part should be selected
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"       # this is formatted: data_location + "name.xml"
) 

def detect_bounding_box(vid):                                           # this function takes a video, outputs faces
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)                  # this 
    faces = face_classifier.detectMultiScale(
        gray_image, 1.1, 5, minSize=(40, 40)
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

<<<<<<< HEAD
while True:
    ret, frame =cap.read()

    faces = detect_bounding_box(frame)  # apply the function we created to the video frame

    cv2.imshow("My Face Detection Project", frame)
      #
=======
# using this tutorial just to demonstrate that the webcam works: 
# https://medium.com/@unknown.underme/opening-webcam-using-opencv-257ac258e217

while True:
    ret, frame = cap.read()

    cv2.imshow('video', frame)
>>>>>>> 61140cbcf85f81f4d5592763b365c9a48d139e56
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()