import cv2
import numpy as np
import time

# Initialize shared data
frame_count = 0
fps = 0
start_time = time.time()
shared_frame = None
shared_pts = None
prev_gray = None
avgx, avgy = 0, 0
tracking = False

def feature_tracking():
    global shared_frame, shared_pts, prev_gray, avgx, avgy, tracking
    lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    while True:
        if shared_frame is not None:
            gray = shared_frame.copy()
            
            if shared_pts is not None and prev_gray is not None:
                prev_pts = shared_pts.copy()
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
                if tracking:
                    tracker = cv2.TrackerKCF_create()
                    success, bbox = tracker.update(frame)
                    if success:
                        bbox = bbox
                    else:
                        tracking = False
                if not tracking:
                    if next_pts is not None and status is not None:
                        good_new = next_pts[status == 1]
                        if good_new.size > 2:
                            avgx, avgy = np.average(good_new, axis=0)
                            good_new = good_new[np.linalg.norm(good_new - np.array([avgx, avgy]), axis=1) < 100]
                            if len(good_new) < 40:
                                roi = gray[int(avgy-100):int(avgy+100), int(avgx-100):int(avgx+100)]
                                new_pts = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=1, blockSize=3)
                                if new_pts is not None:
                                    new_pts[:, :, 0] += avgx-100
                                    new_pts[:, :, 1] += avgy-100
                                    new_pts = new_pts.reshape(-1, 1, 2)
                                    shared_pts = new_pts
                                else:
                                    shared_pts = good_new.reshape(-1, 1, 2) if good_new.size else None
                            else:
                                shared_pts = good_new.reshape(-1, 1, 2) if good_new.size else None
            
            prev_gray = gray.copy()
            shared_frame = None
        else:
            time.sleep(0.0001)

cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking:  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            bbox = (x, y, w, h)
            tracker = cv2.TrackerKCF_create()
            success = tracker.init(frame, bbox)
            if success:
                tracking = True

    shared_frame = gray.copy()

    if shared_pts is not None:
        for pt in shared_pts:
            cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)
    cv2.rectangle(frame, (int(avgx-100), int(avgy-100)), (int(avgx+100), int(avgy+100)), (255, 0, 0), 2)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if (time.time() - start_time) > 1 :
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
