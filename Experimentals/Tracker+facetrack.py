# Taken from opencv
import cv2

# Initialize the video capture object to capture video from the default camera.
cap = cv2.VideoCapture(0)  # 0 for default camera

# Load the Haar cascade XML for face detection, provided by OpenCV.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to detect a face in the frame and return its bounding box.
def get_face_roi(frame):
    # Convert the frame to grayscale since Haar cascades require grayscale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If at least one face is detected, return the bounding box of the first face.
    for (x, y, w, h) in faces:
        return (x, y, w, h)
    return None  # If no face is detected, return None.

# Capture the first frame and get the face bounding box to initialize tracking.
roi = get_face_roi(cap.read()[1])
if roi is not None:
    # If a face is detected, initialize the CSRT tracker.
    tracker = cv2.TrackerCSRT_create()
    tracker.init(cap.read()[1], roi)

# Continuous loop to process each frame from the video stream.
while True:
    # Read a frame from the video stream.
    ret, frame = cap.read()
    
    # If reading the frame failed (e.g., end of video), exit the loop.
    if not ret:
        break

    # If we have a valid ROI (i.e., face bounding box).
    if roi:
        # Update the tracker to get the new bounding box for the face.
        (success, box) = tracker.update(frame)
        
        # If tracking is successful, draw the bounding box around the face.
        if success:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # If tracking failed, try to detect the face again.
            roi = get_face_roi(frame)
            if roi:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, roi)
    else:
        # If no valid ROI, try to detect the face.
        roi = get_face_roi(frame)
        if roi:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, roi)

    # Display the frame with the tracking (and detection) result.
    cv2.imshow("Tracking", frame)
    
    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

