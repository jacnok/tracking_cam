import cv2
import numpy as np

class Rect:
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.ex = None
        self.ey = None
        self.cx = None
        self.cy = None

    def overlap(self, box):
        deltax = self.cx - box.cx
        deltay = self.cy - box.cy
        dist = (deltax * deltax) + (deltay * deltay)
        if dist < (self.w * box.w):
            return True, dist
        return False, dist

    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.ex = x + w
        self.ey = y + h
        self.cx = x + (w / 2)
        self.cy = y + (h / 2)

    def setT(self, Tuple):
        self.x, self.y, self.w, self.h = Tuple
        self.ex = self.x + self.w
        self.ey = self.y + self.h
        self.cx = self.x + (self.w / 2)
        self.cy = self.y + (self.h / 2)

class Person:
    def __init__(self, frame, bbox, tracker_type='KCF'):
        self.tracker_types = {'CSRT': cv2.TrackerCSRT_create, 'KCF': cv2.TrackerKCF_create}
        self.tracker = self.tracker_types[tracker_type]()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.rect = Rect()
        self.rect.setT(bbox)
        self.roi = None
        self.update_roi(frame)
        self.confidence = 2
        self.tracking = True

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = bbox
            self.rect.setT(bbox)
            self.update_roi(frame)
        else:
            self.tracking = False
        return success, bbox

    def unpair(self):
        self.confidence -= 1
        return self.confidence <= 0

    def update_roi(self, frame):
        self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200))

def draw_boxes(frame, boxes):
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

persons = []
frame_idx = 0
face_detection_interval = 10
max_faces = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % face_detection_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        for new_face in faces:
            box = Rect()
            box.setT(new_face)
            for person in persons[:]:
                if person.rect.overlap(box)[0]:
                    person.rect.setT(new_face)
                    person.update_roi(frame)
                    person.confidence = 2
                    person.tracking = True
                    break
            else:  # No break occurred
                persons.append(Person(frame, new_face))

    # Update all trackers
    for person in persons[:]:
        if not person.update(frame)[0] and person.unpair():
            persons.remove(person)

    draw_boxes(frame, [person.bbox for person in persons if person.bbox])

    frame_idx += 1

    if max_faces is None:
        max_faces = frame.shape[1] // 100

    bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")

    # Place each person's face in the bottom panel
    for i, person in enumerate(persons[:max_faces]):
        bottom_panel[:, i*100:(i+1)*100] = person.roi

    frame = np.vstack((frame, bottom_panel))
    cv2.imshow("My Face Detection Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
