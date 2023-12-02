import rect as R
import cv2
import numpy as np
class Person:
    def __init__(self, tracker_type='KCF'):
        # Select the type of tracker
        self.tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'GOTURN':cv2.TrackerGOTURN_create
        }
        self.recentPair=False
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None
        self.rect=R.Rect()
        self.confidence=30
        self.tracking=True

    def init(self, frame, bbox):
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200))  # The resized face image
        self.tracking=True
    def update(self, frame):
        self.recentPair=False
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.set(bbox)
                # self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200)) 
            else:
                self.tracking=False
            return success, bbox
        except:
            self.tracking=False
            return False,bbox
    def unpair(self,flow,frame):
        step=20
        size = 100
        h, w = frame.shape[:2]

        # Optimized boundary calculations
        sx, sy = max(self.rect.cx - size, 0), max(self.rect.cy - size, 0)
        ex, ey = min(self.rect.cx + size, w), min(self.rect.cy + size, h)

        # Grid calculation
        y, x = np.mgrid[sy:ey:step, sx:ex:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # Adjust flow relative to average
        lines = np.vstack([x, y, x + fx*np.abs(fx), y + fy*np.abs(fy)]).T.reshape(-1, 2, 2)
        fx, fy = fx - np.mean(fx), fy - np.mean(fy)

        lines = np.int32(lines + 0.5)
        #  Averaging process
        distances = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        valid_lines = lines[distances > 10]
        if len(valid_lines) > 0:
            avg_position = np.mean(valid_lines[:, 0], axis=0)
            self.confidence=30
            self.rect.cx, self.rect.cy = avg_position
            self.rect.setC(self.rect.cx, self.rect.cy)
        
        
        if len(valid_lines) <=0:
            self.confidence=self.confidence-1
            print(self.confidence)
        if self.confidence<=0:
            return True
        else: 
            return False
    def get_image(self):
        return self.roi