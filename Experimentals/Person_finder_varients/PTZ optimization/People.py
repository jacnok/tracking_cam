import rect as R
import cv2
import numpy as np

lk_params = dict(winSize=(40, 40), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .3)) 
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
        self.confidence=0
        self.tracking=True
        self.color=(0,0,0)
        self.prev_pts = None

    def init(self, frame, bbox):
        self.bbox = bbox
        self.rect.set(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
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
        self.recentPair=False
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.set(bbox)
                if self.confidence!= 30:
                    if self.confidence==5:
                        faces = face_cascade.detectMultiScale(self.roi, 1.7, 4, minSize=(50, 50))
                        if len(faces)==0:
                            faces= profile_face.detectMultiScale(self.roi, 1.7, 4, minSize=(40, 40))
                        if len(faces)<0:
                            self.confidence=self.confidence-40
                            print("no face")
                        else:
                            print("face")
                    self.confidence=self.confidence+1
                self.roi = cv2.resize(frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)], (100, 200)) 
            else:
                self.tracking=False
            return success, bbox
        except:
            self.tracking=False
            return False,bbox
    def unpair(self,flow,frame):
        step=10
        size = 150
        h, w = flow.shape[:2]

        # Optimized boundary calculations
        sx, sy = max(self.rect.cx - size, 0), max(self.rect.cy - size, 0)
        ex, ey = min(self.rect.cx + size, w), min(self.rect.cy + size, h)

        # Grid calculation
        y, x = np.mgrid[sy:ey:step, sx:ex:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        # get the average color at each grid point
        target_color = np.array(self.color).reshape(1, 3)
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
            self.confidence=+5
            avg_position=((self.rect.cx+avg_position[0])/2,(self.rect.cy+avg_position[1])/2)
            self.rect.cx, self.rect.cy = avg_position
            self.rect.setC(self.rect.cx, self.rect.cy)
        
        if len(valid_lines) <=0:
            self.confidence=self.confidence-1
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
        
        if self.confidence<=0:
            return True, valid_lines
        else: 
            return False, valid_lines
    def get_image(self):
        return self.roi
    def unpair2(self,prev_gray,gray):
        if np.array_equal(prev_gray, gray):
            # self.confidence=self.confidence+1
            return False
        
        
        if self.prev_pts is not None and self.confidence>0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.prev_pts, None, **lk_params)
            good_new = next_pts[status == 1]
            # if len(good_new) <4 or self.confidence<10:
            #     good_old = self.prev_pts[status == 1]
            #     displacement = np.sqrt((good_new[:,0] - good_old[:,0])**2 + (good_new[:,1] - good_old[:,1])**2)
            #     significant_movement = displacement > 1
            #     good_new = good_new[significant_movement]
            # self.prev_pts = good_new.reshape(-1, 1, 2)
            if len(good_new) > 0:
                if (len(good_new) > 15):
                    self.confidence=self.confidence+1
                else:
                    self.confidence=self.confidence-1
                # Compute the bounding box coordinates
                x_min, y_min = np.average(good_new, axis=0)
                self.rect.setC(x_min, y_min)
                # cv2.rectangle(frame, (int(x_min-w/2), int(y_min-h/2)), (int(x_min+w/2), int(y_min+h/2)), (255, 0, 0), 2)

                # for new, old in zip(good_new, good_old):
                #     x, y = new.ravel()
                    # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                self.prev_pts = good_new.reshape(-1, 1, 2)
            else:
                return True
        else:
            return True
        return False
    def innitt2(self,frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Spot=frame[int(self.rect.y):int(self.rect.ey), int(self.rect.x):int(self.rect.ex)]
        # Spot=cv2.cvtColor(Spot, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(Spot, maxCorners=100, qualityLevel=.1, minDistance=2)
        if self.prev_pts is not None:
                    self.prev_pts[:, :, 0] += self.rect.x  # Adjust x-coordinates of the points
                    self.prev_pts[:, :, 1] += self.rect.y