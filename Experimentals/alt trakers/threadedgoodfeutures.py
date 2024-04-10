import cv2
import numpy as np
import threading

# Initialize locks for thread-safe operations
frame_lock = threading.Lock()
pts_lock = threading.Lock()

# Initialize shared data
shared_frame = None
shared_pts = None
prev_gray = None

def feature_tracking():
    global shared_frame, shared_pts, prev_gray
    lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        with frame_lock:
            # Only proceed if there is a frame to process
            if shared_frame is None:
                continue  # Skip the rest of the loop if no new frame is available
            
            # Copy the shared frame so we can work on it
            frame = shared_frame.copy()

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        with pts_lock:
            if shared_pts is not None and prev_gray is not None:
                # Calculate optical flow
                prev_pts = shared_pts.copy()
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

                good_new = next_pts[status == 1]
                shared_pts = good_new.reshape(-1, 1, 2)

        prev_gray = gray.copy()

# Start the feature tracking thread
threading.Thread(target=feature_tracking, daemon=True).start()

cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with pts_lock:
        if shared_pts is None or len(shared_pts) < 10:  # Check if face detection is needed
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi = gray[y:y+h, x:x+w]
               
                new_pts = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
                if new_pts is not None:
                    new_pts[:, :, 0] += x
                    new_pts[:, :, 1] += y
                    shared_pts = new_pts

    # Copy the frame to the shared variable
    with frame_lock:
        shared_frame = frame.copy()

    # Drawing points
    if shared_pts is not None:
        for pt in shared_pts:
            cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
