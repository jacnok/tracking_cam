import cv2
import torch
import gc

# Function to release GPU memory
def free_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Load YOLOv5n model (Nano model for less memory usage)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=False)

# Load image
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
persondetected = False
tracker=cv2.TrackerKCF_create()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    clone = cv2.resize(frame, (160,120), interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
    # Perform inference
# take box results and give create a cv2.kcf tracker when person is detected
# update the tracker with the new frame
# if the tracker is lost, delete the tracker
    if persondetected == False:
        print("Detecting persons")
        results = model(clone)
        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1*4), int(y1*4)), (int(x2*4), int(y2*4)), (255, 0, 0), 2)
            if int(cls) == 0:  # Class 0 is person
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(clone, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                persondetected = True
    else:
        ok, bbox = tracker.update(clone)
        if ok:
            p1 = (int(bbox[0])*4, int(bbox[1])*4)
            p2 = (int(bbox[0] + bbox[2])*4, int(bbox[1] + bbox[3])*4)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        else:
            print("Tracker lost")
            persondetected = False
            tracker = None
        
    cv2.imshow('Detected Persons', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # if cv2.waitKey(1) & 0xFF == ord('p'):
    #     persondetected = False
cv2.destroyAllWindows()