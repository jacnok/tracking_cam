import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")


# Initialize previous points and previous gray frame
prev_pts = None
prev_gray = None

# Create a Lucas-Kanade optical flow object
lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, .5)) 
w,h=0,0
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is None:
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Select the first face (assuming only one face is detected)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi =  cv2.resize(gray[y:y+h, x:x+w],(100,200))
            prev_pts = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=.01, minDistance=10)
            if prev_pts is not None:
                prev_pts[:, :, 0] += x  # Adjust x-coordinates of the points
                prev_pts[:, :, 1] += y  # Adjust y-coordinates of the points
    elif prev_gray is not None:
        # Calculate optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        # Select the points that are successfully tracked
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Draw the tracked points and bounding box on the frame
        if len(good_new) > 0:
            # Compute the bounding box coordinates
            x_min, y_min = np.average(good_new, axis=0)
            cv2.rectangle(frame, (int(x_min-w/2), int(y_min-h/2)), (int(x_min+w/2), int(y_min+h/2)), (255, 0, 0), 2)

            for new, old in zip(good_new, good_old):
                x, y = new.ravel()
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Update the previous points
            
            prev_pts = good_new.reshape(-1, 1, 2)
        else:
            # If no good points to track, find a new face
            prev_pts = None

    # Update the previous frame
    prev_gray = gray.copy()

    # Display the frame
    cv2.imshow('Face Tracking', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
