import cv2
import threading
from deepface import DeepFace
# Initialize the Haar Cascade face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_image=cv2.imread("me.jpg")
# Start video capture
cap = cv2.VideoCapture(0)
counter=0
ret, frame = cap.read()
# Variable to keep track of when we find a face to start the GOTURN tracker
face_found = False
face_match=False
def checkface(frame):
    global face_match
    try:
        face_match=DeepFace.verify(frame,known_image.copy())['verified']
    except Exception:
        print("oops")


checkface(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If we haven't found a face yet, look for one
    # if not face_found:
    #     # Convert frame to grayscale for face detection
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #     # If faces are found, take the first one and initialize the tracker with it
        
    #     if len(faces) > 0:
    #         (x, y, w, h) = faces[0]
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #         # threading.Thread()
    #         face_found=True
    if counter % 40 == 0:
        threading.Thread(target=checkface,args=(frame.copy(),)).start()
    
    if face_match:
        cv2.putText(frame,"Match",(28,458),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0,3))
    counter+=1
    # Display the resulting frame
    cv2.imshow('FaceRECO', frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        checkface(frame)
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()