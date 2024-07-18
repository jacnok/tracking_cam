import rect as R
import cv2
import numpy as np
import cv2
import torch
import time
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
lk_params = dict(winSize=(100, 100), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, .9)) 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
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
        self.confidence=4
        self.tracking=True
        self.color=(0,0,0)
        self.prev_pts = None
        self.shirttrack=None
        self.shirttracking=False
        self.roi=None
        self.alive=None

    def init(self, frame, bbox):
        self.alive=time.time()
        self.confidence=20
        self.prev_pts = None
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
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
                if time.time()-self.alive>10:
                    self.tracking=False
                # self.roi = cv2.resize(frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)], (100, 200)) 
            else:
                self.tracking=False
            return success
        except Exception as e:
            print(e)
            self.tracking=False
            return False
    def get_image(self):
        return self.roi
    def unpair(self,prev_gray,gray,notgray):
        
        if self.prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.prev_pts, None, **lk_params)
            
            good_new = next_pts[status == 1]
            # Draw the tracked points and bounding box on the frame
            if len(good_new) > 2:
            # Compute the bounding box coordinates
                x_avg, y_avg = np.average(good_new, axis=0)
                self.rect.setC(x_avg, y_avg)
                good_new = good_new[np.linalg.norm(good_new - np.array([x_avg, y_avg]), axis=1) < 100]
                self.prev_pts = good_new.reshape(-1, 1, 2) if good_new.size else None
            if len(good_new) < 30:
                self.confidence-=1
                roi = gray[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)]
                new_pts = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=4, blockSize=7)
                # good_new = good_new.reshape(-1, 1, 2)
                            
                if new_pts is not None:
                    new_pts[:, :, 0] += self.rect.x
                    new_pts[:, :, 1] += self.rect.y
                    new_pts = new_pts.reshape(-1, 1, 2)
                    new_pts = np.vstack((new_pts, good_new.reshape(-1, 1, 2)))
                    self.prev_pts = new_pts
                else:
                    self.prev_pts = good_new.reshape(-1, 1, 2) if good_new.size else None
        if self.prev_pts is None:
             Spot=gray[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)]
             self.prev_pts = cv2.goodFeaturesToTrack(Spot, maxCorners=200, qualityLevel=.1, minDistance=4, blockSize=7)
             if self.prev_pts is not None:
                    self.prev_pts[:, :, 0] += self.rect.x  # Adjust x-coordinates of the points
                    self.prev_pts[:, :, 1] += self.rect.y
        
        if self.confidence<1:
            return True
        return False
    def altnn(self,notgray,scalex,scaley):
        if not self.shirttracking:
            results = model(notgray)
            for box in results.xyxy[0]:
                
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0:  # Class 0 is person
                    if self.rect.cx > x1 and self.rect.cx < x2:
                    # Ensure coordinates are integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Initialize the tracker on the smaller frame
                        self.shirttrack = cv2.TrackerKCF_create()
                        self.shirttrack.init(notgray, (x1, y1, x2 - x1, y2 - y1))
                        self.shirttracking = True
                        print ("found shirt")
                        break
            if not self.shirttracking:
                self.confidence-=1
        
        else:
            ok, bbox = self.shirttrack.update(notgray)
            if ok:
                # Scale coordinates back to original frame size
                x1, y1, w, h = bbox
                cx=int((x1*scalex)+(w*scalex/2))
                cy=int((y1*scaley)+(h*scaley/2))
                self.rect.setC(cx,cy)
            else:
                print("failed")
                self.shirttracking = False
                self.shirttrack = None
        print(self.confidence)        
        if self.confidence<1 and (time.time()-self.alive>3):
            
            return True
        return False
    
def bodysearch(notgray,scalex,y):
            results = model(notgray)
            found=False
            
            for box in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0 and not found:  # Class 0 is person
                    # Ensure coordinates are integers
                        found=True
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w=x2-x1
                        # Initialize the tracker on the smaller frame
                        return (int((x1+(w/2))*scalex),int( y),100,150)            
            if not found:
                    return None
              