import cv2
import numpy as np

        
tracking=False
# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
ret, frame = cap.read()

frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
# Initialize previous points and previous gray frame
prev_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sx, sy, w, h = 0, 0, 0, 0
cx,cy=0,0
color=(0,0,0)
confidence=20
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not tracking:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5, minSize=(30, 30))
        # Select the first face (assuming only one face is detected)
        if len(faces) > 0:
            (sx, sy, w, h) = faces[0]
            roi = gray[sy:sy+h, sx:sx+w]
            cx,cy=sx+w//2,sy+h//2
            color=(np.mean(roi[...,0]),np.mean(roi[...,1]),np.mean(roi[...,2]))
            tracking=True
            confidence=20
            cv2.rectangle(frame, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (0,255,0), 2)
    elif prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 4, 1, 1, .3, 0)
        step=40
        size = 100
        h, w = flow.shape[:2]
        
        # Optimized boundary calculations
        ssx, ssy = max(cx - size, 0), max(cy - size, 0)
        ex, ey = min(cx + size, w), min(cy + size, h)
        # Grid calculation
        frame = cv2.rectangle(frame, (int(ssx), int(ssy)), (int(ex), int(ey)), (0, 0, 255), 2)
        y, x = np.mgrid[ssy:ey:step, ssx:ex:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        # get the average color at each grid point
        target_color = np.array(color).reshape(1, 3)
        colorpoints = frame[y, x]  # shape (N, 3)
        color_diff = np.linalg.norm(colorpoints - target_color, axis=1) 
         # print(colorpoints)
        color_match_threshold = 100  # Define a threshold for color matching
        # Adjust flow relative to average
        lines = np.vstack([x, y, x + fx*np.abs(fx), y + fy*np.abs(fy)]).T.reshape(-1, 2, 2)
        fx, fy = fx - np.mean(fx), fy - np.mean(fy)
        lines = np.int32(lines + 0.5)
        #  Averaging process
        distances = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        valid_lines = lines[(distances > 20) & (color_diff < color_match_threshold)]
       
        # valid_lines = lines[(distances > 20)]
        if len(valid_lines) > 0:
            avg_position = np.mean(valid_lines[:, 0], axis=0)
            confidence=+5
            avg_position=((cx+avg_position[0])/2,(cy+avg_position[1])/2)

            cx, cy = avg_position
            cv2.rectangle(frame, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (0,255,0), 2)
        for points in valid_lines:
            cv2.line(frame, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
        if len(valid_lines) <=0:
            confidence=confidence-1
             # print(self.confidence)
            # expeirmental
                # if self.confidence<=20:
                #     self.tracker=cv2.TrackerKCF_create()
                #     self.tracker.init(frame, (self.rect.x,self.rect.y,self.rect.w,self.rect.h))
                #     try:
                #         self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200))  # The resized face image
                #     except Exception as e:
                #         print(e)
                #         print (self.rect.x,self.rect.y,self.rect.ex,self.rect.ey)    
                #     self.tracking=True
                #     #get the average color of the face
                #     self.color=(np.mean(self.roi[...,0]),np.mean(self.roi[...,1]),np.mean(self.roi[...,2]))
                #     self.bbox=None
            
        if confidence<=0:
            tracking=False
    
    cv2.imshow('Face Tracking', frame)
    prev_gray = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()