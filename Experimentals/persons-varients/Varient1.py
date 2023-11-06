import cv2
import numpy as np

class Rect:
    def __init__(self, bbox=(0, 0, 0, 0)):
        self.setT(bbox)

    def overlap(self, box):
        # Check if the distance between centers is less than the average width
        dist = np.hypot(self.cx - box.cx, self.cy - box.cy)
        threshold = (self.w + box.w) / 2
        return dist < threshold, dist

    def setT(self, bbox):
        self.x, self.y, self.w, self.h = bbox
        self.ex = self.x + self.w
        self.ey = self.y + self.h
        self.cx = self.x + (self.w / 2)
        self.cy = self.y + (self.h / 2)

class Person:
    tracker_types = {
        'CSRT': cv2.TrackerCSRT_create,
        'KCF': cv2.TrackerKCF_create
    }

    def __init__(self, tracker_type='KCF'):
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None
        self.rect = Rect()
        self.confidence = 2
        self.tracking = True
        self.roi = None

    def init(self, frame, bbox):
        self.bbox = bbox
        self.rect.setT(bbox)
        self.tracker.init(frame, bbox)
        self.update_roi(frame)
        self.confidence = 2

    def update(self, frame):
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
                self.rect.setT(bbox)
                self.update_roi(frame)
                return success, bbox
        except cv2.error as e:
            print(f"OpenCV error during tracking update: {e}")
            self.tracking = False
            # Determine here if you need to reinitialize the tracker or not
        return False, None

    def unpair(self):
        self.confidence -= 1
        return self.confidence <= 0

    def update_roi(self, frame):
        if self.bbox is not None:
            x, y, w, h = [int(v) for v in self.bbox]
            x_end = min(x + w, frame.shape[1])
            y_end = min(y + h, frame.shape[0])
            x = max(0, x)
            y = max(0, y)
            self.roi = cv2.resize(frame[y:y_end, x:x_end], (100, 200))
    def get_image(self):
        return self.roi
    
def draw_boxes(frame, persons):
    for person in persons:
        if person.bbox is not None:
            x, y, w, h = [int(v) for v in person.bbox]
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

        for new_face in faces:
            box.setT(new_face)
            matched = False

            for person in persons[:]:
                if person.rect.overlap(box)[0]:
                    if not matched:
                        person.rect.setT(new_face)
                        person.init(frame, new_face)  # Reinitialize the person
                        matched = True
                    else:
                        persons.remove(person)

            if not matched:
                new_person = Person('KCF')
                new_person.init(frame, new_face)
                persons.append(new_person)
        # print("escape")
    # Update all trackers and remove the ones that failed
    for person in persons:
        if person.update(frame)[0]==False:
            if person.unpair():
                persons.remove(person)
    # Draw bounding boxes from trackers
    
    draw_boxes(frame, persons)
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