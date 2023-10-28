import cv2

# Initialize the video capture object to capture video from the default camera.
cap = cv2.VideoCapture(0)  # 0 for default camera
track=False
# Load the Haar cascade XML for face detection, provided by OpenCV.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to detect a face in the frame and return its bounding box.
def get_face_roi(frame):
    # Convert the frame to grayscale since Haar cascades require grayscale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If at least one face is detected, return the bounding box of the first face.
    if len(faces)>0:
        return faces
    return []  # If no face is detected, return None.

ret, frame = cap.read()
last_rois=get_face_roi(frame)
# Capture the first frame and get the face bounding box to initialize tracking.
trackers=[]

# Continuous loop to process each frame from the video stream.
while True:
    # Read a frame from the video stream.
    
    ret, frame = cap.read()
    rois= get_face_roi(frame)
    
    if len(last_rois) !=len(rois):
        if len(last_rois)>len(rois) and len(last_rois)>0:
            if track==False:
                trackers=[]
                for L_roi in last_rois:
                    cv2.rectangle(frame,L_roi,(0,0,255),1)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame,L_roi)
                    trackers.append(tracker)
                    print("red")
                    track=True
        elif len(last_rois)<len(rois):
            for (x, y, w, h) in rois:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                print("green")
                track=False
    else:
        for (x, y, w, h) in rois:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            track=False
            print("yeelow?")
    if track:
                for tracker in trackers:
                    (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(a) for a in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
                    print("blue")
    #  cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    last_rois=rois
    

    # Display the frame with the tracking (and detection) result.
    cv2.imshow("Tracking", frame)
    
    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

