import cv2
import numpy as np
import rect as R
import People as P
from pynput import keyboard
import MotorController as M
import threading
import time
print(cv2.__version__)

mc=M.Mcontrol()
alttracking=False
def draw_boxes(frame, peaple):
    for p in peaple:
        if p.tracking:
            color = (0, 255, 0)
            # color = p.color
        else:
            color = (0, 0, 255)
            for person in persons:
                if person.prev_pts is not None:
                    for pts in zip(person.prev_pts):
                        # print(pts)
                        x, y = map(int, pts[0][0].ravel())  # Convert x and y to integers
                        # print(x, y)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw the rectangle
        cv2.rectangle(frame, (int(p.rect.x), int(p.rect.y)), (int(p.rect.ex), int(p.rect.ey)), color, 2)

key_held=False
def on_press(key):
    global key_pressed, key_held
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            if mc.keypressed(key_pressed,key_held)==False:
                controls(key_pressed)
                key_held=True
    except AttributeError:
        pass
endthread=True
lines=[]
def track():
    global persons, frame, endthread,cap,resized,prev_gray, gray,lines,alttracking
    while not endthread:
        try:
            if not cap.isOpened():
                break
            calcflow=False
            if len(persons)>0:
                # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]
                if(resized or not sizechange):
                    
                    for person in persons:
                        if person.update(frame)[0]==False or alttracking:
                            if alttracking:
                                person.tracking=False
                            if calcflow==False:
                                    # flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 4, 1, 1, .3, 0)
                                    person.innitt2(gray)
                                    calcflow = True
                            # lost,lines=person.unpair(flow,frame)
                            lost=person.unpair2(prev_gray,gray)
                            if lost:
                                persons.remove(person)
                                print("got rid of him")
            else:
                mc.none()
                break
            if endthread:
                break
        except Exception as e:
            print(e)
            break
        
    prev_gray = gray.copy()
    endthread=True
def track2():
    global persons, frame, endthread,cap,resized,prev_gray, gray,lines
    try:
            calcflow=False
            if len(persons)>0:
                # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]
               
                    
                    for person in persons:
                        person.tracking=False

                        if calcflow==False:
                                    # flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 4, 1, 1, .3, 0)
                                    person.innitt2(gray)
                                    calcflow = True
                            # lost,lines=person.unpair(flow,frame)
                        
                        lost=person.unpair2(prev_gray,gray)
                        if lost:
                            persons.remove(person)
                            print("got rid of him")
    except Exception as e:
        print("error in track2")
        print(e)


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
presetcalled=False
try:
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
    # cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
    sizechange=True
except:
    cap = cv2.VideoCapture(0)
    sizechange=False
# cap = cv2.VideoCapture(0)
persons = []
frame_idx = 0
face_detection_interval = 1  # Detect faces every 10 frames
box=R.Rect()
selected=0

ret, frame = cap.read()
frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = gray
def trackmovment(head,frame,boundx,boundy):
    global detect , key_held,mc
    movment=False
    if detect and False:
        if key_held==False:
            if (head.x<boundx):
                sensitiv=abs((head.x-boundx)/head.w)*18
                if sensitiv>14:
                    sensitiv=14
                elif sensitiv<1:
                    sensitiv=1
                mc.Senitivityx=int(sensitiv)
                cv2.line(frame, (boundx,0), (boundx,frame.shape[0]), (0,255,0), 2)
                mc.L=1
                movment=True
            elif (head.ex>frame.shape[1]-boundx):
                sensitiv=abs((head.ex-(frame.shape[1]-boundx))/head.w)*18
                if sensitiv>14:
                    sensitiv=14
                elif sensitiv<1:
                    sensitiv=1
                mc.Senitivityx=int(sensitiv)
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
            # if (head.w<(frame.shape[1]/8)):
            #     mc.zoom=1
            #     movment=True
            # elif (head.w>(frame.shape[1]/1.5)):
            #     mc.zoom=-1
            #     movment=True
            # else:
            #     mc.zoom=0
            
            if movment==False:
                mc.none()
            else:
                mc.write()
boundx=200
boundy=125
showbounds=False
def controls(key_pressed):
    global boundy,boundx,frame,showbounds,persons,selected, detect,alttracking
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
    if key_pressed=="b":
        if selected>0 and len(persons)>1:
            selected-=1
    elif key_pressed=="n":
        if (selected+1)<len(persons):
            selected+=1
    if key_pressed=="r":
        while len(persons)>0:
            for person in persons:
                persons.remove(person)
        mc.stopmove()
    if key_pressed=="e":
        if(detect):
            detect= False
        else:
            detect=True
        print(detect)
    if key_pressed=="t":
        alttracking= not alttracking
        print(alttracking)
    # print(boundx)
    # print(boundy)

while True:
    resized=False
    ret, frame = cap.read()
    if presetcalled:
        if detect==False:
            if delay+5<time.time():
                detect=True
                presetcalled=False
    if ret:
        if sizechange:
            frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
            # frame = cv2.convertScaleAbs(frame, alpha=2, beta=0)# Adjust this value, >1 to increase contrast, 0-1 to decrease/to fix rasism
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized=True
        if frame_idx % face_detection_interval == 0 and detect and not key_held and not alttracking:
            # Convert frame to grayscale
            
            # Resize the frame for faster detection (scale down)
            scale_factor=0.9
            small_gray=cv2.resize(gray,None,fx=scale_factor,fy=scale_factor)
            # Detect faces on the smaller image
            if len(persons)==0 or face_detection_interval<10:
                faces = face_cascade.detectMultiScale(small_gray, 1.3, 7, minSize=(50, 50))
                
                if len(faces)==0:
                    faces= profile_face.detectMultiScale(small_gray, 1.5, 8, minSize=(40, 40))
                    
                
            if endthread==True:
                if len(persons)>0:
                    endthread=False
                    threading.Thread(target=track).start()
            faces=[((int(x/scale_factor)),(int(y/scale_factor)),(int(w/scale_factor)),(int(h/scale_factor))) for (x,y,w,h) in faces]
            if len(faces)>0 and face_detection_interval>20:
                faces = face_cascade.detectMultiScale(gray,1.1, 6, minSize=(50, 50))
                print("big")
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
                            if dist>200 or not person.tracking:
                                person.init(frame, new_face)
                    if person.rect.xmatch(box):
                        matched=True
                if not matched:
                    old_len=len(persons)
                    new_person = P.Person(tracker_type='KCF')
                    new_person.init(frame, new_face)
                    persons.append(new_person)
                    if old_len==0 and len(persons)>0:
                        endthread=False
                        threading.Thread(target=track).start()
        for peaple in persons:
            if not peaple.tracking: #be sure to remove the false
                face_detection_interval=2
                break
            elif len(persons)<0:
                face_detection_interval=1
            else:
                face_detection_interval=30
        ## Update all trackers and remove the ones that have failed
        # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()
        
        # Draw bounding boxes for all tracked persons
        # for person in persons:
        #     if person is not None:
        #         if person.tracking:
        #            draw_boxes(frame,person.bbox)
            #         else:
            #             draw_boxes(frame,person.bbox ,(0,0,255))
        draw_boxes(frame, [person for person in persons if person.bbox is not None])
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
        # Process face images to display at the bottom panelrdsd
        max_faces = frame.shape[1] // 100
        bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")
        for i, person in enumerate(persons[:max_faces]):
            bottom_panel[:, i * 100:(i + 1) * 100] = person.get_image()
            if i==selected:            
                cv2.rectangle(bottom_panel,(i*100,0),((i + 1) * 100,200),(0,255,0),4)

        # Stack the main frame and the bottom panel


        frame = np.vstack((frame, bottom_panel))
        # Display the frame
        side_panel = np.zeros((frame.shape[0], 250, 3), dtype="uint8")
       
        texts=["Press 'e' key to enable",
               "and disable face",
               "recogonition",
               "Press WASD keys to move",
               "Press 'f' or 'g' to", 
               "adjust horizontal bounds",
               "Press 'c' or 'v' to",
               "adjust vertical bounds",
               "Press 'b' or 'n' to",
               "switch sqaures",
                "Press 'r' to reset all",
                "detected sqaures"
        ]
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_color=(255,255,255)
        for i,text in enumerate(texts):
            cv2.putText(side_panel,text,(10,20+(i*25)),font,.6,font_color,1)
        # cv2.putText(side_panel,text1,(10,20),font,.5,font_color,1)
        frame = np.hstack((frame, side_panel))
        cv2.imshow("My Face Detection Project", frame)

        # Break the loop if 'q' is pressed
    else:
        print("not ret")
        cap = cv2.VideoCapture(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    prev_gray = gray.copy()
# Release resources
endthread=True
cap.release()
cv2.destroyAllWindows()
mc.close()
listener.stop()


def callpreset(preset):
    global persons,delay,presetcalled,detect
    while len(persons)>0:
            for person in persons:
                persons.remove(person)
    delay=time.time()
    mc.preset(preset)
    presetcalled=True
    detect=False    
    pass