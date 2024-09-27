import rect as R
import cv2
import numpy as np
import torch
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)

# Define LK parameters
lk_params = dict(winSize=(100, 100), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, .9)) 

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

class Person:
    def __init__(self, tracker_type='KCF'):
        self.tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create,
            'GOTURN':cv2.TrackerGOTURN_create
        }
        self.recentPair=False
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None
        self.rect=R.Rect()
        self.confidence=4
        self.tracking=True
        self.shirttrack=None
        self.shirttracking=False
        self.roi=None
        self.alive=None
        self.oldx=None
        self.movment=0

    def init(self, frame, bbox):
        self.alive=time.time()
        self.confidence=20
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.oldx=self.rect.cx
        try:
            self.roi = cv2.resize(frame[max(0, int(self.rect.y)):int(self.rect.ey), max(0, int(self.rect.x)):int(self.rect.ex)], (100, 200))
        except Exception as e:
            self.tracking=False
            print(e)
            print (self.rect.x,self.rect.y,self.rect.ex,self.rect.ey)    
        self.tracking=True
    def update(self, frame):
        
        self.recentPair=False
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.set(bbox)
                self.movment+=abs(self.oldx-self.rect.cx)
                if time.time()-self.alive>3:
                    if self.movment>10:
                        self.tracking=False
                # self.roi = cv2.resize(frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)], (100, 200)) 
            else:
                self.tracking=False
            self.oldx=self.rect.cx
            return success
        except Exception as e:
            print(e)
            self.tracking=False
            return False
    def get_image(self):
        return self.roi
    def altnn(self,notgray,scalex,scaley):
        if not self.shirttracking:
            results = model(notgray)
            for box in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0 and self.rect.cx > x1 and self.rect.cx < x2:  # Class 0 is person
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    self.shirttrack = cv2.TrackerKCF_create()
                    self.shirttrack.init(notgray, (x1, y1, x2 - x1, y2 - y1))
                    self.shirttracking = True
                    print("found shirt")
                    break
            if not self.shirttracking:
                self.confidence -= 1
        else:
            ok, bbox = self.shirttrack.update(notgray)
            if ok:
                x1, y1, w, h = bbox
                cx, cy = int((x1 * scalex) + (w * scalex / 2)), int((y1 * scaley) + (h * scaley / 2))
                self.rect.setC(cx, cy)
            else:
                print("failed")
                self.shirttracking = False
                self.shirttrack = None
        print(self.confidence)
        return self.confidence < 1 and (time.time() - self.alive > 3)

def bodysearch(notgray, scalex, y):
    results = model(notgray)
    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0:  # Class 0 is person
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w = x2 - x1
            return int((x1 + (w / 2)) * scalex), int(y), 100, 150
    return None
