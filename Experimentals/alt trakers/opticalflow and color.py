import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to read from the webcam
# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.001, minDistance=2, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Color for drawing
color = (0, 255, 0)

# Initialize some variables
face_box = None
points = None
prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if points is None:
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_box = (x, y, x + w, y + h)
            # Select good features within the face region
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            points = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
            prev_gray = gray.copy()
            avg_x, avg_y = np.mean(points[:, 0], axis=0).astype(int)
            continue

    if prev_gray is not None:
        # Calculate optical flow
        nextPts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)
        # Select good points
        good_new = nextPts[status == 1]
        good_old = points[status == 1]

        # Draw points
        for pt in good_new:
            a, b = pt.ravel()
            cv2.circle(frame, (int(a),int(b)), 5, color, -1)

        # Update points to track
        
        good_new = good_new[(good_new[:, 0] >= avg_x - w//2) & (good_new[:, 0] <= avg_x + w//2) &
                                (good_new[:, 1] >= avg_y - h//2) & (good_new[:, 1] <= avg_y + h//2)]
        points = good_new.reshape(-1, 1, 2)

        # Calculate bounding box based on average of points
        avg_x, avg_y = np.mean(points[:, 0], axis=0).astype(int)
        x, y, x2, y2 = face_box
        w, h = x2 - x, y2 - y
        cv2.rectangle(frame, (avg_x - w//2, avg_y - h//2), (avg_x + w//2, avg_y + h//2), (255, 0, 0), 2)

        # Detect color changes within the bounding box to add points
        if prev_gray is not None and len(points) < 100:
            roi_prev = prev_gray[avg_y-100 - h//2:avg_y +100+ h//2, avg_x -100- w//2:avg_x +100+ w//2]
            roi_curr = gray[avg_y-100 - h//2:avg_y +100+ h//2, avg_x -100- w//2:avg_x +100+ w//2]
            diff = cv2.absdiff(roi_curr, roi_prev)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(diff)
            if max_val > 25:  # Threshold for significant color change
                new_point = np.array([[[avg_x - w//2 + max_loc[0], avg_y - h//2 + max_loc[1]]]], dtype=np.float32)
                points = np.vstack([points, new_point])
        
    # Update previous frame and previous points
    prev_gray = gray.copy()

    # Show the frame
    cv2.imshow('Face Tracking with Feature and Color Change Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
