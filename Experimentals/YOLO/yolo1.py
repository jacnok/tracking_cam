import cv2
import torch

# Load YOLOv5s model (use a smaller model if available)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)

# Load video from webcam
cap = cv2.VideoCapture(0)
person_detected = False
tracker = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for inference
    small_frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_LINEAR)
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    frame_height, frame_width = frame.shape[:2]

    if not person_detected:
        results = model(small_frame)
        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Class 0 is person
                # Ensure coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Initialize the tracker on the smaller frame
                tracker = cv2.TrackerKCF_create()
                tracker.init(small_frame, (x1, y1, x2 - x1, y2 - y1))
                person_detected = True
                break
    else:
        ok, bbox = tracker.update(small_frame)
        if ok:
            # Scale coordinates back to original frame size
            x1, y1, w, h = bbox
            p1 = (x1*4, y1*4)
            p2 = ((x1 + w)*4, (y1 + h)*4)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        else:
            person_detected = False
            tracker = None
    
    # Display the frame
    cv2.imshow('Detected Persons', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
