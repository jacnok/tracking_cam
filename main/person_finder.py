import cv2
import numpy as np
import rect as R
import People as P
from pynput import keyboard
import MotorController as M
import threading

mc=M.Mcontrol()

def draw_boxes(frame, boxes):
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
key_held=False
def on_press(key):
    global key_pressed, key_held
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            if mc.keypressed(key_pressed)==False:
                controls(key_pressed)
                key_held=True
    except AttributeError:
        pass
endthread=False
def track():
    global persons, frame, endthread,cap
    while True:
        try:
            # if not cap.isOpened():
            #     break
            if len(persons)>0:
                # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]
                if(resized):
                    for person in persons:
                        if person.update(frame)[0]==False:
                            if person.unpair():
                                persons.remove(person)
                                print("got rid of him")
            else:
                mc.none()
                break
            if endthread:
                break
        except:
            print(Exception)
            break
        


def on_release(key):
    global key_pressed , key_held
    try:
        if key.char and key.char == key_pressed:
           
            key_pressed = None  # Reset the key state
            mc.none()
            key_held=False
    except AttributeError:
        pass

# Start listener for key press and release
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
detect=True
resized=False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upperbody=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
sizechange=True
# try:
#     cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
#     sizechange=True
# except:
cap = cv2.VideoCapture(0)
persons = []
frame_idx = 0
face_detection_interval = 1  # Detect faces every 10 frames
box=R.Rect()
selected=0

ret, frame = cap.read()
# frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
def trackmovment(head,frame,boundx,boundy):
    global detect
    movment=False
    if detect:
        if key_held==False:
            if (head.x<boundx):
                cv2.line(frame, (boundx,0), (boundx,frame.shape[0]), (0,255,0), 2)
                mc.L=1
                movment=True
            elif (head.ex>frame.shape[1]-boundx):
                cv2.line(frame, (frame.shape[1]-boundx,0), (frame.shape[1]-boundx,frame.shape[0]), (0,255,0), 2)
                mc.r=1
                movment=True
            else:
                mc.r=0
                mc.L=0
            if (head.y<int(boundy-(boundy/2))):
                cv2.line(frame, (0,int(boundy-(boundy/2))), (frame.shape[1],int(boundy-(boundy/2))), (0,255,0), 2)
                mc.u=1
                movment=True
            elif(head.ey>frame.shape[0]-int(boundy+(boundy/2))):
                cv2.line(frame, (0,frame.shape[0]-(int(boundy+(boundy/2)))), (frame.shape[1],frame.shape[0]-int(boundy+(boundy/2))), (0,255,0), 2)
                mc.d=1
                movment=True
            else:
                mc.d=0
                mc.u=0
            
            if movment==False:
                mc.none()
            else:
                mc.write()
boundx=200
boundy=125
showbounds=False
def controls(key_pressed):
    global boundy,boundx,frame,showbounds,persons,selected, detect
    showbounds=False
    if key_pressed=="f":
        if(boundx<int(frame.shape[1]/2)):
            showbounds=True
            boundx+=25
    elif key_pressed=="g":
        if(boundx>0):
            showbounds=True
            boundx-=25
    if key_pressed=="c":
        if(boundy<int(frame.shape[0]/2)):
            showbounds=True
            boundy+=25
    elif key_pressed=="v":
        if(boundy>0):
            showbounds=True
            boundy-=25
    if key_pressed=="z":
        if selected>0 and len(persons)>1:
            selected-=1
    elif key_pressed=="x":
        if (selected+1)<len(persons):
            selected+=1
    if key_pressed=="r":
        while len(persons)>0:
            for person in persons:
                persons.remove(person)
    if key_pressed=="e":
        if(detect):
            detect= False
        else:
            detect=True
        print(detect)
    print(boundx)
    print(boundy)

while True:
    resized=False
    ret, frame = cap.read()

    if ret:
        if sizechange:
            frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
        resized=True
        
        bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")
        if len(persons)<0:
            face_detection_interval=1
        elif face_detection_interval==1:
            face_detection_interval=30
        if frame_idx % face_detection_interval == 0 and detect:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame for faster detection (scale down)
            scale_factor = 0.9  # Example scale factor
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)

            # Detect faces on the smaller image
            faces = face_cascade.detectMultiScale(small_gray, 1.1, 4, minSize=(30, 30))
            if len(faces)==0 and persons==0:
                faces= profile_face.detectMultiScale(small_gray, 1.4, 4, minSize=(30, 30))

            # Adjust detection coordinates to match the original frame size
            faces = [(int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)) for (x, y, w, h) in faces]

            for (x, y, w, h) in faces:
                
                # cv2.rectangle(frame,(x-1,y-1),(x+w+1,y+h+1),(255,0,0),1)
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
                            if dist>2000:  
                                person.init(frame, new_face)
            
                if not matched:
                    old_len=len(persons)
                    new_person = P.Person(tracker_type='CSRT')
                    new_person.init(frame, new_face)
                    persons.append(new_person)
                    if old_len==0 and len(persons)>0:
                        threading.Thread(target=track).start()
        
        ## Update all trackers and remove the ones that have failed
        # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]
        
        
        # Draw bounding boxes for all tracked persons
        draw_boxes(frame, [person.bbox for person in persons if person.bbox is not None])

        # showbounds
        if showbounds==True:     
            cv2.line(frame, (boundx,0), (boundx,frame.shape[0]), (0,255,0), 2)  
            cv2.line(frame, (frame.shape[1]-boundx,0), (frame.shape[1]-boundx,frame.shape[0]), (0,255,0), 2)
            cv2.line(frame, (0,int(boundy-(boundy/2))), (frame.shape[1],int(boundy-(boundy/2))), (0,255,0), 2)
            cv2.line(frame, (0,frame.shape[0]-(int(boundy+(boundy/2)))), (frame.shape[1],frame.shape[0]-int(boundy+(boundy/2))), (0,255,0), 2)
            if frame_idx % 20 == 0:
                showbounds=False
        # Increment frame index
        frame_idx += 1
        if len(persons)>0:
            if (selected+1)>len(persons):
                selected=0
            trackmovment(persons[selected].rect,frame,boundx,boundy)
        # Process face images to display at the bottom panel
        max_faces = frame.shape[1] // 100
        for i, person in enumerate(persons[:max_faces]):
            bottom_panel[:, i * 100:(i + 1) * 100] = person.get_image()
            if i==selected:            
                cv2.rectangle(bottom_panel,(i*100,0),((i + 1) * 100,200),(0,255,0),4)

        # Stack the main frame and the bottom panel
        frame = np.vstack((frame, bottom_panel))

        # Display the frame
        cv2.imshow("My Face Detection Project", frame)

        # Break the loop if 'q' is pressed
    else:
        print("not ret")
        cap = cv2.VideoCapture(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release resources
endthread=True
cap.release()
cv2.destroyAllWindows()
mc.close()
listener.stop()