import cv2
import numpy as np

class Rect:
    def __init__(self):
        self.x=None
        self.y=None
        self.w=None
        self.h=None
        self.ex=None
        self.ey=None
        self.cx=None
        self.cy=None
        # self.area=w*h
    def overlap(self,box):
        deltax=self.cx-box.cx
        deltay=self.cy-box.cy
        dist=(deltax*deltax)+(deltay*deltay)
        if dist<((self.w*box.w)):
            return True , dist
        else:
            print("{0:.2f}>{1:.2f},{2:.2f},{3:.2f}".format(dist,(self.w*box.w),self.w,box.w))
            return False,dist
    def set(self,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.ex=x+w
        self.ey=y+h
        self.cx=x+(w/2)
        self.cy=y+(h/2)
    def setT(self,Tuple):
        self.x=Tuple[0]
        self.y=Tuple[1]
        self.w=Tuple[2]
        self.h=Tuple[3]
        self.ex=x+w
        self.ey=y+h
        self.cx=x+(w/2)
        self.cy=y+(h/2)

class Person:
    def __init__(self, tracker_type='KCF'):
        # Select the type of tracker
        self.tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create
        }
        self.recentPair=False
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None
        self.rect=Rect()
        self.confidence=2
        self.tracking=True

    def init(self, frame, bbox):
        self.bbox = bbox
        self.rect.setT(bbox)
        self.tracker=cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200))  # The resized face image
        self.tracking=True
        self.confidence=2
    def update(self, frame):
        self.recentPair=False
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.setT(bbox)
            else:
                self.tracking=False
            return success, bbox
        except:
            print(frame.shape)
            return False,bbox
    def unpair(self):
        self.confidence=self.confidence-1
        # print(self.confidence)
        if self.confidence<=0:
            return True
        else: 
            return False
    def get_image(self):
        return self.roi

def draw_boxes(frame, boxes):
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

persons = []
frame_idx = 0
face_detection_interval = 10  # Detect faces every 10 frames
box=Rect()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")

    if frame_idx % face_detection_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            new_face = (x, y, w, h)
            box.setT(new_face)
            matched=False
            # Check for overlap with existing faces
            for person in persons[:]:
                if matched:
                    if person.rect.overlap(box)[0]:
                        print("deleting")
                        persons.remove(person)
                else:
                    matched,dist=person.rect.overlap(box)
                    if matched:
                        person.rect.set( x, y, w, h)
                        person.recentPair=True
                        if dist>100000:
                            person.init(frame, new_face)
            # print(matched)
            if not matched:
                new_person = Person(tracker_type='KCF')
                new_person.init(frame, new_face)
                persons.append(new_person)
        # print("escape")
    # Update all trackers and remove the ones that failed
    for person in persons:
        if person.update(frame)[0]==False:
            if person.unpair():
                persons.remove(person)
    # Draw bounding boxes from trackers
    
    draw_boxes(frame, [person.bbox for person in persons if person.bbox is not None])
    frame_idx += 1
    max_faces = frame.shape[1] // 100

    # Place each person's face in the bottom panel up to the maximum number of faces
    for i, person in enumerate(persons[:max_faces]):
        roi = person.get_image()
        bottom_panel[:, i*100:(i+1)*100] = roi

    
    # Stack the main frame and the bottom panel
    frame = np.vstack((frame, bottom_panel))

    # Display the resulting frame
    cv2.imshow("My Face Detection Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(100) & 0xFF == ord('c'):
    #    for person in persons:
    #        print ("remove")
    #        persons.remove(person)
cap.release()
cv2.destroyAllWindows()