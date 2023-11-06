import cv2
import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

class Person:
    def __init__(self, tracker_type='KCF'):
        # Select the type of tracker
        self.tracker_types = {
            'CSRT': cv2.TrackerCSRT_create,
            'KCF': cv2.TrackerKCF_create
        }
        self.tracker = self.tracker_types[tracker_type]()
        self.bbox = None

    def init(self, frame, bbox):
        self.bbox = bbox
        self.tracker.init(frame, bbox)

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = bbox
        return success, bbox

def draw_boxes(frame, boxes):
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

persons = []
frame_idx = 0
face_detection_interval = 10  # Detect faces every 10 frames


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % face_detection_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            new_face = (x, y, w, h)
            # Check for overlap with existing faces
            if not any(get_iou(new_face, person.bbox) > 0.3 for person in persons):  # IoU threshold
                new_person = Person(tracker_type='KCF')
                new_person.init(frame, new_face)
                persons.append(new_person)
                print(type(new_face))

    # Update all trackers and remove the ones that failed
    persons = [person for person in persons if person.update(frame)[0]]
    # Draw bounding boxes from trackers
    draw_boxes(frame, [person.bbox for person in persons if person.bbox is not None])

    cv2.imshow("My Face Detection Project", frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
