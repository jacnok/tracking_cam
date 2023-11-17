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
        self.Name="N/A"
        self.recentPair=False
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None
        self.rect=R.Rect()
        self.confidence=2
        self.tracking=True

    def init(self, frame, bbox):
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200))  # The resized face image
        if not self.Name=="N/A":    
            self.roi = cv2.putText(self.roi, self.Name, (20,180), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv2.LINE_AA) 
        self.tracking=True
        self.confidence=2
    def namechange(self,thename):
        if self.Name=="N/A":
            self.Name=thename
            self.roi = cv2.putText(self.roi, self.Name, (20,180), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv2.LINE_AA) 
    
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
            # print(frame.shape)
            return False,bbox
    def unpair(self):
        self.confidence=self.confidence-1
        if self.confidence<=0:
            return True
        else: 
            return False
    def get_image(self):
        return self.roi