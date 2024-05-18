import rect as R
import cv2
import numpy as np
import cv2

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
        self.confidence=2
        self.tracking=True
        self.color=(0,0,0)
        self.prev_pts = None
        self.shirttrack=None
        self.roi=None

    def init(self, frame, bbox):
        print("found him")
        self.confidence=10
        self.prev_pts = None
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        # print(self.rect.x,self.rect.y,self.rect.ex,self.rect.ey)
        try:
            self.roi = cv2.resize(frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)], (100, 200))  # The resized face image
        except Exception as e:
            self.tracking=False
            print(e)
            print (self.rect.x,self.rect.y,self.rect.ex,self.rect.ey)    
        self.tracking=True
        #get the average color of the face
        # self.color=(np.mean(self.roi[...,0]),np.mean(self.roi[...,1]),np.mean(self.roi[...,2]))
    def update(self, frame):
        # print(self.tracking)
        # if len(frame.shape) == 3:
        #     height, width, channels = frame.shape
        #     print("The frame has {} channels.".format(channels))
        # else:
        #     print("The frame is likely grayscale and has a single channel.")
        self.recentPair=False
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.set(bbox)
                
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
    def unpair(self,prev_gray,gray):
            
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
    def innitt2(self,frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        Spot=frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)]
        # self.shirttrack=cv2.TrackerKCF_create()
        # self.shirttrack.init(frame, ())
        # Spot=cv2.cvtColor(Spot, cv2.COLOR_BGR2GRAY)
        # self.prev_pts = cv2.goodFeaturesToTrack(Spot, maxCorners=200, qualityLevel=.1, minDistance=2)
        height, width = Spot.shape[:2]

# Create an empty mask
        mask = np.zeros_like(Spot)

# Determine the center region of the mask
        left, top = width // 4, height // 4
        right, bottom = width * 3 // 4, height * 3 // 4

# Fill the center region of the mask with a higher value
        mask = cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

# Use the mask in goodFeaturesToTrack
        self.prev_pts = cv2.goodFeaturesToTrack(Spot, maxCorners=200, qualityLevel=.1, minDistance=2, blockSize=7)
        if self.prev_pts is not None:
                    self.prev_pts[:, :, 0] += self.rect.x  # Adjust x-coordinates of the points
                    self.prev_pts[:, :, 1] += self.rect.y
    