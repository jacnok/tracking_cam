import cv2
import numpy as np

# imported python modules as per YT video: 
# https://www.youtube.com/watch?v=rKcwcARdg9M

cap = cv2.VideoCapture(0)

while True:
    ret, frame =cap.read()
    image = np.zeros(frame.shape,np.uint8)

    cv2.imshow('frame',image)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()