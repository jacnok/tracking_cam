import cv2

# Initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the GOTURN tracker
tracker = cv2.TrackerGOTURN_create()

# Start video capture
cap = cv2.VideoCapture(0)

# Variable to keep track of when we find a face to start the GOTURN tracker
face_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If we haven't found a face yet, look for one
    if not face_found:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If faces are found, take the first one and initialize the tracker with it
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker.init(frame, (x, y, w, h))
            face_found = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Otherwise, update the tracker and draw the tracking rectangle
    else:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            face_found = False  # If tracking failed, try to detect the face again

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()