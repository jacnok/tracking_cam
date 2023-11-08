import cv2
import numpy as np

class Rect:
    """ A class to manage the rectangle attributes of a bounding box. """
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.ex = None  # End x (x + width)
        self.ey = None  # End y (y + height)
        self.cx = None  # Center x
        self.cy = None  # Center y

    def overlap(self, box: 'Rect') -> tuple[bool, float]:
        """ Calculate if there's an overlap with another box based on distance. """
        deltax = self.cx - box.cx
        deltay = self.cy - box.cy
        dist = (deltax ** 2) + (deltay ** 2)
        return (dist < (self.w * box.w)), dist

    def set(self, bbox_tuple: tuple[int, int, int, int]):
        """ Update rectangle coordinates and derived metrics. """
        (self.x, self.y, self.w, self.h) = bbox_tuple
        self.ex = self.x + self.w
        self.ey = self.y + self.h
        self.cx = self.x + (self.w/2)
        self.cy = self.y + (self.h/2)

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
        self.rect.set(bbox)
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
                self.rect.set(bbox)
                self.roi = cv2.resize(frame[self.rect.y:self.rect.ey, self.rect.x:self.rect.ex], (100, 200)) 
            else:
                self.tracking=False
            return success, bbox
        except:
            print(frame.shape)
            return False,bbox
    def unpair(self):
        self.confidence=self.confidence-1
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
face_detection_interval = 30  # Detect faces every 10 frames
box=Rect()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")

    if frame_idx % face_detection_interval == 0:
         # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame for faster detection (scale down)
        scale_factor = 0.75  # Example scale factor
        small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)

        # Detect faces on the smaller image
        faces = face_cascade.detectMultiScale(small_gray, 1.1, 4, minSize=(30, 30))

        # Adjust detection coordinates to match the original frame size
        faces = [(int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)) for (x, y, w, h) in faces]

        for (x, y, w, h) in faces:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            new_face = (x, y, w, h)
            box.set(new_face)
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
                        person.rect.set( new_face)
                        person.recentPair=True
                        if dist>100000:
                            person.init(frame, new_face)
            # print(matched)
            if not matched:
                new_person = Person(tracker_type='CSRT')
                new_person.init(frame, new_face)
                persons.append(new_person)
        # print("escape")
    
    ## Update all trackers and remove the ones that have failed
    persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]

    # Draw bounding boxes for all tracked persons
    draw_boxes(frame, [person.bbox for person in persons if person.bbox is not None])

    # Increment frame index
    frame_idx += 1

    # Process face images to display at the bottom panel
    max_faces = frame.shape[1] // 100
    for i, person in enumerate(persons[:max_faces]):
        bottom_panel[:, i * 100:(i + 1) * 100] = person.get_image()

    # Stack the main frame and the bottom panel
    frame = np.vstack((frame, bottom_panel))

    # Display the frame
    cv2.imshow("My Face Detection Project", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()