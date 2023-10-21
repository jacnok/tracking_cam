import cv2
import numpy as np

# imported python modules as per YT video: 
# https://www.youtube.com/watch?v=rKcwcARdg9M

cap = cv2.VideoCapture(0)

# using this tutorial just to demonstrate that the webcam works: 
# https://medium.com/@unknown.underme/opening-webcam-using-opencv-257ac258e217

while True:
    ret, frame = cap.read()

    cv2.imshow('video', frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()