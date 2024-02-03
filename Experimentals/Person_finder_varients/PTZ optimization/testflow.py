import cv2
import rect as R
import numpy as np
class Person:
    def __init__(self):
        self.bbox = None
        self.rect=R.Rect()
        self.confidence=0
        self.tracking=True
        self.color=(0,0,0)
        self.prev_pts = None
    def unpair2(self,prev_gray,gray):
        # if np.array_equal(prev_gray, gray):
        #     return True
        
        lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .001)) 
        if self.prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.prev_pts, None, **lk_params)
            good_new = next_pts[status == 1]
            # self.prev_pts = good_new.reshape(-1, 1, 2)
            if len(good_new) > 0:
                # Compute the bounding box coordinates
                x_min, y_min = np.average(good_new, axis=0)
                self.rect.setC(x_min, y_min)
                # cv2.rectangle(frame, (int(x_min-w/2), int(y_min-h/2)), (int(x_min+w/2), int(y_min+h/2)), (255, 0, 0), 2)

                # for new, old in zip(good_new, good_old):
                #     x, y = new.ravel()
                    # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                self.prev_pts = good_new.reshape(-1, 1, 2)
            else:
                self.confidence=0
                return True
        else:
            return True
    def innitt2(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(frame, maxCorners=100, qualityLevel=.1, minDistance=10)
        if self.prev_pts is not None:
                    self.prev_pts[:, :, 0] += self.rect.x  # Adjust x-coordinates of the points
                    self.prev_pts[:, :, 1] += self.rect.y

cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a Person instance
person = Person()

# Initialize a previous frame variable
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0 and person.prev_pts is None:
        # Initialize tracking for the first detected face
        (x, y, w, h) = faces[0] # Assuming R.Rect takes x, y, width, height
        roi = frame[y:y+h, x:x+w]
        person.rect.set((x, y, w, h))
        person.innitt2(roi)  # Initialize tracking points

    # Track the person in the current frame
    if person.prev_pts is not None:
        person.unpair2(prev_gray, gray)

    # Draw the bounding box
    if person.prev_pts is not None:
        cv2.rectangle(frame, (int(person.rect.x), int(person.rect.y)), (int(person.rect.ex), int(person.rect.ey)), (0,255,0), 2)
        for pts in zip(person.prev_pts):
                        # print(pts)
                        x, y = map(int, pts[0][0].ravel())  # Convert x and y to integers
                        # print(x, y)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


    # Display the output
    cv2.imshow('Person', frame)

    # Update previous frame
    prev_gray = gray.copy()

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()