import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

# Initialize previous points and previous gray frame for both face and shirt
prev_pts_face = None
prev_pts_shirt = None
prev_gray = None
face_bbox_initial = None
shirt_bbox_initial = None

# Create a Lucas-Kanade optical flow object
lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, .5))

def get_shirt_region(face_bbox, frame):
    """Estimate the shirt region based on the face bounding box."""
    x, y, w, h = face_bbox
    shirt_top = y + h + 10  # Start a bit below the face to avoid overlap
    shirt_height = int(h * 1.5)  # Estimate the shirt height
    shirt_bottom = min(shirt_top + shirt_height, frame.shape[0])
    return (x, shirt_top, w, shirt_bottom - shirt_top)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts_face is None:
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Assuming the first detected face
            face_bbox_initial = faces[0]
            (x, y, w, h) = face_bbox_initial

            # Initialize face tracking points
            face_roi = gray[y:y+h, x:x+w]
            prev_pts_face = cv2.goodFeaturesToTrack(face_roi, maxCorners=100, qualityLevel=.01, minDistance=10)
            if prev_pts_face is not None:
                prev_pts_face[:, :, 0] += x  # Adjust x-coordinates of the points
                prev_pts_face[:, :, 1] += y  # Adjust y-coordinates of the points

            # Estimate and initialize shirt tracking points
            shirt_bbox_initial = get_shirt_region(face_bbox_initial, frame)
            shirt_x, shirt_y, shirt_w, shirt_h = shirt_bbox_initial
            shirt_roi = gray[shirt_y:shirt_y+shirt_h, shirt_x:shirt_x+shirt_w]
            prev_pts_shirt = cv2.goodFeaturesToTrack(shirt_roi, maxCorners=100, qualityLevel=.01, minDistance=10)
            if prev_pts_shirt is not None:
                prev_pts_shirt[:, :, 0] += shirt_x  # Adjust x-coordinates of the points
                prev_pts_shirt[:, :, 1] += shirt_y  # Adjust y-coordinates of the points

    elif prev_gray is not None:
        # Calculate optical flow and track points for both face and shirt
        next_pts_face, status_face, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_face, None, **lk_params)
        good_new_face = next_pts_face[status_face == 1]
        prev_pts_face = good_new_face.reshape(-1, 1, 2)

        next_pts_shirt, status_shirt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_shirt, None, **lk_params)
        good_new_shirt = next_pts_shirt[status_shirt == 1]
        prev_pts_shirt = good_new_shirt.reshape(-1, 1, 2)

        # Draw fixed-size bounding boxes and tracking points
        (x, y, w, h) = face_bbox_initial
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Face box in green
        for pt in good_new_face:
            cv2.circle(frame, ((pt[0]), (pt[1])), 3, (0, 255, 0), -1)  # Face tracking points

        (x, y, w, h) = shirt_bbox_initial
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Shirt box in red
        # for pt in good_new_shirt:
        #     cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 3, (0, 0, 255), -1)  # Shirt tracking points

    # Update the previous frame
    prev_gray = gray.copy()

    # Display the frame
    cv2.imshow('Face and Shirt Tracking', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
