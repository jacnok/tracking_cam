import cv2
import numpy as np
import rect as R
import People as P
from pynput import keyboard
import MotorController as M
import threading
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
print(cv2.__version__)
from flask import Flask,request


boundx=200
boundy=25
delay=0 #cando / OPTIONAL -> use delay to slowdown searching for faces (when not in use)
adjustbounds=True
interupt=False
ptzmode=True
app = Flask(__name__)
if ptzmode:
    mc=M.Mcontrol("192.168.20.202")
else:
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
    global key_pressed, key_held,interupt,delay
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            if mc.keypressed(key_pressed,key_held)==False:
                controls(key_pressed)
                key_held=True
            interupt=True
            delay=time.time()
    except AttributeError:
        pass
endthread=True
lines=[]


shared_frame = None
def track():
    global persons, shared_frame, endthread,cap,resized,prev_gray, lines,alttracking
    index=0
    while not endthread:
        if shared_frame is None:
                time.sleep(0.0001)
        else:  
            try:
                if not cap.isOpened():
                    break
                if len(persons)>0:
                    # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()]
                    if(resized or not sizechange):
                        
                        for person in persons:
                            # shared_frame = cv2.cvtColor(shared_frame, cv2.COLOR_BGR2GRAY)
                            if person.update(shared_frame)==False or alttracking:
                                # if shared_frame is None:
                                #     time.sleep(0.0001)
                                #     continue 
                                if alttracking:
                                    person.tracking=False
                                    # person.innitt2(gray)
                                    
                                lost=person.unpair(prev_gray,gray)

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
            shared_frame = None
    endthread=True


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
# listener = keyboard.Listener(on_press=on_press, on_release=on_release) # disable for now
# listener.start()

detect=True
resized=False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upperbody=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
sizechange=True
presetcalled=False
# try:
#     # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testlvl2.mp4")
# cap = cv2.VideoCapture(r"C:\Users\cherr\Documents\Processing\resoarces\testpaster.mp4")
#     sizechange=True
# except:
#     cap = cv2.VideoCapture(0)
#     sizechange=False
cap = cv2.VideoCapture(0)
persons = []
frame_idx = 0
face_detection_interval = 1  # Detect faces every 10 frames
box=R.Rect()
selected=0
nn=True
scale_factor2 = 0.25  # Example: Reduce size by half

# Create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=160, margin=120, keep_all=True)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def callpreset(preset):
    global persons,delay,presetcalled,detect,boundx
    while len(persons)>0:
            for person in persons:
                persons.remove(person)
    delay=time.time()
    mc.preset(preset)
    presetcalled=True
    if preset==1:
        boundx=100
    else:
        boundx=175
    print("preset called!!!")
    detect=False    
    pass
def stream_deck_command(command):
    global mc,interupt,delay,persons,detect,boundx,boundy,showbounds
    interupt=True
    print(command)
    delay=time.time()
    if command=="up":
        delay+=10
        mc.u=1
    elif command=="down":
        delay+=10
        mc.d=1
    elif command=="left":
        delay+=10
        mc.L=1
        mc.Senitivityx=10
    elif command=="right":
        delay+=10
        mc.Senitivityx=10
        mc.r=1
    elif command=="stopmove":
        mc.Senitivityx=2
        mc.none()
        interupt=False
    elif command=="zoomin":
        delay+=10
        mc.zoom=1
    elif command=="zoomout":
        delay+=10
        mc.zoom=-1
    elif command=="reset":
        while len(persons)>0:
            for person in persons:
                persons.remove(person)
    elif command=="stop":
        if(detect):
            detect= False
        else:
            detect=True
    if ptzmode:
        mc.Senitivityx=1
    mc.write()
    if command=="boundsx+":
        showbounds=True
        boundx+=25
        interupt=False
    elif command=="boundsx-":
        showbounds=True
        boundx-=25
        interupt=False
    if command=="boundsy+":
        showbounds=True
        boundy+=25
        interupt=False
    elif command=="boundsy-":
        showbounds=True
        boundy-=25
        interupt=False
@app.route('/callpreset', methods=['POST'])
def handle_callpreset():
    data = request.json
    preset = data.get("preset")
    move = data.get("move")
    print(move)
    if preset is not None or move is not None:
        # Run the callpreset function in a new thread to not block the Flask server
        if preset is not None:
            threading.Thread(target=callpreset, args=(preset,)).start()
        elif move is not None:
            threading.Thread(target=stream_deck_command,args=(move,)).start()
        return {"status": "success", "message": f"Preset {preset} called"}
    else:
        return {"status": "error", "message": "Missing preset parameter"}, 400
def run_flask_app():
    app.run(debug=False, port=5000)
    time.sleep(1)
# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app,daemon=True)
flask_thread.start()
ret, frame = cap.read()
frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = gray
shared_frame = frame.copy()
def trackmovment(head,frame,boundx,boundy):
    global detect , key_held,mc
    movment=False
    try:
        if detect:
            if key_held==False:
                if (head.y<int(boundy-(3*boundy/4))):
                    cv2.line(frame, (0,int(boundy-(3*boundy/4))), (frame.shape[1],int(boundy-(3*boundy/4))), (0,255,0), 2)
                    mc.u=1
                    sensitiv=3
                    movment=True
                elif(head.ey>frame.shape[0]-int(boundy+(3*boundy/4))):
                    cv2.line(frame, (0,frame.shape[0]-(int(boundy+(3*boundy/4)))), (frame.shape[1],frame.shape[0]-int(boundy+(3*boundy/4))), (0,255,0), 2)
                    mc.d=1
                    sensitiv=3
                    movment=True
                else:
                    mc.d=0
                    mc.u=0
                if (head.x<boundx):
                    sensitiv=int(abs((head.x-boundx)/head.w)*18)
                    if sensitiv>14:
                        sensitiv=14
                    elif sensitiv<4:
                        sensitiv=3
                    if ptzmode:
                        if sensitiv>12:
                            sensitiv=2
                    mc.Senitivityx=int(sensitiv)
                    cv2.line(frame, (boundx,0), (boundx,frame.shape[0]), (0,255,0), 2)
                    mc.L=1
                    movment=True
                elif (head.ex>frame.shape[1]-boundx):
                    sensitiv=int(abs((head.ex-(frame.shape[1]-boundx))/head.w)*18)
                    if sensitiv>14:
                        sensitiv=14
                    elif sensitiv<4:
                        sensitiv=3
                    if ptzmode:
                        if sensitiv>12:
                            sensitiv=2
                    mc.Senitivityx=int(sensitiv)
                    cv2.line(frame, (frame.shape[1]-boundx,0), (frame.shape[1]-boundx,frame.shape[0]), (0,255,0), 2)
                    mc.r=1
                    movment=True
                else:
                    mc.r=0
                    mc.L=0
                # if (head.w<(frame.shape[1]/8)): #zoom
                #     mc.zoom=1
                #     movment=True
                # elif (head.w>(frame.shape[1]/1.5)):
                #     mc.zoom=-1
                #     movment=True
                # else:
                #     mc.zoom=0
                if movment==False and ptzmode ==False:  #micromovent
                    microturn=head.cx-(frame.shape[1]/2)
                    if abs(microturn)>40:
                        mc.Senitivityx=2
                    else:
                        mc.Senitivityx=1
                    if microturn>15:
                        mc.r=1
                        mc.L=0
                        movment=True
                    elif microturn<-15:
                        mc.L=1
                        mc.r=0
                        movment=True
                if movment==False:
                    mc.none()
                else:
                    mc.write()
    except Exception as e:
        print(e)
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
        # mc.stopmove()
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
screenwidth=1920
cv2.namedWindow("My Face Detection Project",cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("My Face Detection Project", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.moveWindow("My Face Detection Project", screenwidth, 0)
while True:
    resized=False
    ret, frame = cap.read()
    if presetcalled:
        if detect==False:
            if delay+.9<time.time():
                detect=True
                presetcalled=False
                face_detection_interval=1
    if ret:
        if sizechange:
            frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=0.9)# Adjust this value, >1 to increase contrast, 0-1 to decrease/to fix rasism
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shared_frame = frame.copy()
        resized=True
        if frame_idx % face_detection_interval == 0 and detect and not key_held and not alttracking:
            # Convert frame to grayscale
            # Resize the frame for faster detection (scale down)
            scale_factor=0.9
            small_gray=cv2.resize(gray,None,fx=scale_factor,fy=scale_factor)
            # Detect faces on the smaller image
            if len(persons)==0 or face_detection_interval<10:
                faces = face_cascade.detectMultiScale(small_gray, 1.3, 7, minSize=(100, 100))
                if len(faces)==0:
                    faces= profile_face.detectMultiScale(small_gray, 1.2, 4, minSize=(80, 80))
            if endthread==True:
                if len(persons)>0:
                    endthread=False
                    threading.Thread(target=track).start()

            #show a red box around the face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame,(int(x/scale_factor),int(y/scale_factor)),(int((x+w)/scale_factor),int((y+h)/scale_factor)),(0,0,255),1)

            faces=[((int(x/scale_factor)),(int(y/scale_factor)),(int(w/scale_factor)),(int(h/scale_factor))) for (x,y,w,h) in faces]
            if face_detection_interval>20 and len(persons)>0:
                faces = face_cascade.detectMultiScale(gray,1.5, 8, minSize=(100, 100))
                if len(faces)>0: 
                    print("found "+str(len(faces)))
            
            if nn and len(faces)==0:
                if face_detection_interval==2 or face_detection_interval>20:
                        small_frame = cv2.resize(frame, None, fx=scale_factor2, fy=scale_factor2, interpolation=cv2.INTER_LINEAR)
                        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        boxes, _ = mtcnn.detect(small_frame_rgb)
                        if boxes is not None:
                            faces=[((int(x1/scale_factor2)),(int(y1/scale_factor2)),(int((x2-x1)/scale_factor2)),(int((y2-y1)/scale_factor2))) for (x1,y1,x2,y2) in boxes]
                
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame,(x,y-1),(x+w+1,y+h+1),(255,0,0),10)
                
                new_face = (x, y, w, h)
                box.set(new_face)
                matched=False
                # Check for overlap with existing faces
                for person in persons[:]:
                    if matched:
                        if person.rect.overlap(box)[0]:
                            persons.remove(person)
                    else:
                        matched,dist=person.rect.overlap(box)
                        print(dist)
                        if matched:
                            person.rect.set( new_face)
                            person.recentPair=True
                            
                            if dist<10000 or not person.tracking:
                                print("rescan")
                                person.init(frame, new_face)
                    if person.rect.xmatch(box):
                        matched=True
                if not matched:
                    old_len=len(persons)
                    adjustbounds=True
                    new_person = P.Person(tracker_type='KCF')
                    new_person.init(frame, new_face)
                    persons.append(new_person)
                    if old_len==0 and len(persons)>0:
                        endthread=False
                        
                        threading.Thread(target=track).start()
            if face_detection_interval==2:                
                    for peaple in persons:
                        if not peaple.tracking: #be sure to remove the false
                            peaple.confidence-=1


        if len(persons)==0:
            face_detection_interval=1
        else:
            face_detection_interval=80
        for peaple in persons:
            if not peaple.tracking: #be sure to remove the false
                face_detection_interval=2
                break
        ## Update all trackers and remove the ones that have failed
        # persons[:] = [person for person in persons if person.update(frame)[0] or not person.unpair()
        
        # Draw bounding boxes for all tracked persons
        # for person in persons:
        #     if person is not None:
        #         if person.tracking:
        draw_boxes(frame, [person for person in persons if person.bbox is not None])
        if showbounds==True:     
            cv2.line(frame, (boundx,0), (boundx,frame.shape[0]), (0,255,0), 2)  
            cv2.line(frame, (frame.shape[1]-boundx,0), (frame.shape[1]-boundx,frame.shape[0]), (0,255,0), 2)
            cv2.line(frame, (0,int(boundy-(3*boundy/4))), (frame.shape[1],int(boundy-(3*boundy/4))), (0,255,0), 2)
            cv2.line(frame, (0,frame.shape[0]-(int(boundy+(3*boundy/4)))), (frame.shape[1],frame.shape[0]-int(boundy+(3*boundy/4))), (0,255,0), 2)
            if frame_idx % 20 == 0:
                showbounds=False
        # Increment frame index
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL        
        text="Tracking off"
        if  not detect or key_held:
            cv2.putText(frame,text,(20,100),font,1,(0,0,255),2)
        # Increment frame index
        frame_idx += 1
        if len(persons)>0:
            # if adjustbounds:
            #     boundx=int(abs(persons[selected].rect.w*2-frame.shape[1])/2)
            #     adjustbounds=False
            if (selected+1)>len(persons):
                selected=0
            
            if interupt==False:
                trackmovment(persons[selected].rect,frame,boundx,boundy)
            else:
                if delay+1<time.time():
                    interupt=False
        # Process face images to display at the bottom panelrdsd
        max_faces = frame.shape[0] // 200
        
        # bottom_panel_null = np.zeros((100, frame.shape[1], 3), dtype="uint8")
        
        
        side_panel_new = np.zeros((frame.shape[0], 100, 3), dtype="uint8")
        
        
        try:
            for i, person in enumerate(persons[:max_faces]):
                if person.get_image() is not None:
                    person_image = person.get_image()
                        # Ensure the image is resized to the expected shape (200, 100, 3)
                    if person_image.shape != (200, 100, 3):
                        person_image = cv2.resize(person_image, (100, 200))

                    side_panel_new[(i) * 200 : (i + 1) * 200, :] = person_image
                    # print("new person added!")
                    
                    if i==selected:            
                        cv2.rectangle(side_panel_new,
                                      
                                      (0, (i) * 200),             # horizontal, start point at (0, 0), where (x, y) 
                                      (100, (i + 1) * 200),           # vertical, start point at (0, 0), where (x, y)
                                      (0, 255, 0),              # green
                                      4)                        # stroke size
                else :
                    persons.remove(person)
                    side_panel_new[:, i * 100:(i + 1) * 100] = np.zeros((200, 100, 3), dtype="uint8")
                    
        except Exception as e:
            print(e)            

        # Stack the main frame and the bottom panel


        # frame = np.vstack((bottom_panel_null, frame, bottom_panel_null))
        
        
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
        
        frame = np.hstack((frame, side_panel_new))
        if frame_idx>1000:
            frame_idx=0
        
        # added debug screen sizes
        church_small_screen = [(2100,1080), -1920, -1000]
        # debug_screen_jacnok = [(2100,1080), -1920, -1000]
        
        # TODO: set frame_position back to church_small_screen_2[0] for prod
        frame_size = church_small_screen[0]
        
        if frame is None or frame.size == 0:
            raise ValueError("The frame is empty or not valid.")   
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)

        # cv2.setWindowProperty("My Face Detection Project", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # TODO: set move_window_x and move_window_y back to church_small_screen_2[1] and church_small_screen_2[2]  for prod
        move_window_x = church_small_screen[1]
        move_window_y = church_small_screen[2]
        
        cv2.imshow("My Face Detection Project", frame)
        cv2.moveWindow("My Face Detection Project", move_window_x, move_window_y)
        # cv2.resizeWindow ("My Face Detection Project",   cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # ax = ("prop vs. reg: {}, {}").format(cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # print(ax)
        # print(cv2.WND_PROP_FULLSCREEN)
        
        # cv2.namedWindow("My Face Detection Project",cv2.WINDOW_FULLSCREEN)
        # cv2.setWindowProperty("My Face Detection Project", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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


# listener.stop()
