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
        if dist<((self.w)*(box.w)):
            return True
        else:
            # print(dist)
            # print ("{0:.2f}-{1:.2f}>{2:.2f},{3:.2f}".format(box.x,box.ex,self.x,self.ex))
            return False
    def set(self,x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.ex=x+w
        self.ey=y+h
        self.cx=x+(w/2)
        self.cy=y+(h/2)

class Person:
    def __init__(self, x, y, w, h, frame):
        self.roi = cv2.resize(frame[y:y+h, x:x+w], (100, 200))  # The resized face image
        self.rect=Rect()
        self.rect.set(x,y,w,h)
        self.tracking=False
        self.recentPair=False
        self.confidence=100
        self.tracker = cv2.TrackerCSRT_create()
    def update(self):
        self.recentPair=False
    def Track(self,frame):
        self.recentPair=False
        (success, box) = self.tracker.update(frame)
        if success:
            (x, y, w, h) = [int(a) for a in box]
            self.rect.set(x,y,w,h)
            return cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

    def StartTrack(self,frame):
        self.tracking=True
        roi=(self.rect.x, self.rect.y, self.rect.w, self.rect.h)
        self.tracker.init(frame,roi)
    def get_image(self):
        return self.roi
    def unpair(self):
        self.confidence=self.confidence-1
        if self.confidence<=0:
            return True
        else: 
            return False

def detect_faces(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4, minSize=(30, 30))
    return faces

cap = cv2.VideoCapture(0)  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

persons = []
box=Rect()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create the bottom panel
    bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")

    # Detect faces in the current frame
    detected_faces = detect_faces(frame)
    if len(persons)==0:
        for (x, y, w, h) in detected_faces:
            persons.append(Person(x, y, w, h, frame))
            print("adding")
    elif len(detected_faces)>len(persons):
        for (x, y, w, h) in detected_faces:
            persons.append(Person(x, y, w, h, frame))

            # print("adding more")                     turn off tracking
    elif len(detected_faces)<len(persons):
        for (x, y, w, h) in detected_faces:
            matched=False
            box.set(x,y,w,h)
            for person in persons[:]:
                if matched:
                    if person.rect.overlap(box):
                        if person.unpair():
                            persons.remove(person)
                else:
                    matched=person.rect.overlap(box)
                    if matched:
                        person.rect.set( x, y, w, h)
                        person.recentPair=True
        for person in persons[:]:
            if person.recentPair==False:
                person.StartTrack(frame)
    if len(persons)==len(detected_faces):

        for person in persons[:]:
            if person.recentPair==False:
                for (x, y, w, h) in detected_faces:
                    box.set(x,y,w,h)
                    matched = False
                    matched=person.rect.overlap(box)
                    if matched:
                        person.rect.set( x, y, w, h)
                        person.tracking=False
                        person.recentPair=True
                        break
                if person.recentPair==False:
                    if person.unpair():
                        persons.remove(person)
    # print(len(persons))
    for person in persons:
        if person.tracking==False:
            person.update()
        else:
            person.Track(frame)

    for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
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
    if cv2.waitKey(1) & 0xFF == ord('c'):
       for person in persons:
           print ("remove")
           persons.remove(person)
cap.release()
cv2.destroyAllWindows()
