import cv2
import ATEMController
import numpy as np
import rect as R
import People as P
from pynput import keyboard
import MotorController as M
import threading
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import argparse
from flask import Flask, request
import traceback

print(cv2.__version__)

# Flask app initialization
app = Flask(__name__)

# Initial global variables
boundx = 100
boundy = 25
delay = 0
interupt = False
ptzmode = False
debug = True
autocut = False
direct = False
quitprogram =False

# Function to handle key press events
def on_press(key):
    global key_pressed, key_held, interupt, delay
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            print(mc.extract_position())
            if mc.keypressed(key_pressed, key_held) == False:
                controls(key_pressed)
                key_held = True
            interupt = True
            delay = time.time()
    except AttributeError:
        pass

# Function to handle key release events
def on_release(key):
    global key_pressed, key_held
    try:
        if key.char and key.char == key_pressed:
            key_pressed = None  # Reset the key state
            mc.none()
            key_held = False
    except AttributeError:
        pass

# Function to draw bounding boxes around detected people
def draw_boxes(frame, people):
    for p in people:
        if p.tracking:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            for person in persons:
                if person.prev_pts is not None:
                    for pts in zip(person.prev_pts):
                        x, y = map(int, pts[0][0].ravel())  # Convert x and y to integers
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Draw the rectangle
        cv2.rectangle(frame, (int(p.rect.x), int(p.rect.y)), (int(p.rect.ex), int(p.rect.ey)), color, 2)

# Function to track movement
def track():
    global persons, shared_frame, endthread, cap, resized, prev_gray, alttracking
    while not endthread:
        if shared_frame is None:
            time.sleep(0.0001)
        else:
            try:
                if not cap.isOpened():
                    break
                if len(persons) > 0:
                    if resized or not sizechange:
                        for person in persons:
                            if person.update(shared_frame) == False or alttracking:
                                frame_height, frame_width = shared_frame.shape[:2]
                                notgray = cv2.resize(shared_frame, (160, 120), interpolation=cv2.INTER_CUBIC)
                                if alttracking:
                                    person.tracking = False
                                lost = person.altnn(notgray, int(frame_width / 160), int(frame_height / 120))
                                if lost:
                                    persons.remove(person)
                                    print("Removed person")
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
    endthread = True

# Function to track movement based on bounding box and head position
def trackmovement(head, frame, boundx, boundy, tracking):
    global detect, key_held, mc
    movement = False
    try:
        if detect:
            if not key_held:
                if (head.y < int(boundy - (3 * boundy / 4))) and tracking:
                    cv2.line(frame, (0, int(boundy - (3 * boundy / 4))), (frame.shape[1], int(boundy - (3 * boundy / 4))), (0, 255, 0), 2)
                    mc.u = 1
                    sensitiv = 3
                    movement = True
                elif (head.ey > frame.shape[0] - int(boundy + (3 * boundy / 4))) and tracking:
                    cv2.line(frame, (0, frame.shape[0] - int(boundy + (3 * boundy / 4))), (frame.shape[1], frame.shape[0] - int(boundy + (3 * boundy / 4))), (0, 255, 0), 2)
                    mc.d = 1
                    sensitiv = 3
                    movement = True
                else:
                    mc.d = 0
                    mc.u = 0
                if head.x < boundx:
                    sensitiv = int(abs((head.x - boundx) / head.w) * 18)
                    if sensitiv > 12:
                        sensitiv = 14
                    elif sensitiv < 4:
                        sensitiv = 3
                    if ptzmode:
                        if sensitiv > 12:
                            sensitiv = 2
                    mc.Senitivityx = int(sensitiv)
                    cv2.line(frame, (boundx, 0), (boundx, frame.shape[0]), (0, 255, 0), 2)
                    mc.L = 1
                    movement = True
                elif head.ex > frame.shape[1] - boundx:
                    sensitiv = int(abs((head.ex - (frame.shape[1] - boundx)) / head.w) * 18)
                    if sensitiv > 12:
                        sensitiv = 14
                    elif sensitiv < 4:
                        sensitiv = 3
                    if ptzmode:
                        if sensitiv > 12:
                            sensitiv = 2
                    mc.Senitivityx = int(sensitiv)
                    cv2.line(frame, (frame.shape[1] - boundx, 0), (frame.shape[1] - boundx, frame.shape[0]), (0, 255, 0), 2)
                    mc.r = 1
                    movement = True
                else:
                    mc.r = 0
                    mc.L = 0
                if not movement and not ptzmode:  # micromovement
                    microturn = head.cx - (frame.shape[1] / 2)
                    if abs(microturn) > 40:
                        mc.Senitivityx = 2
                    else:
                        mc.Senitivityx = 1
                    if microturn > 15:
                        mc.r = 1
                        mc.L = 0
                        movement = True
                    elif microturn < -15:
                        mc.L = 1
                        mc.r = 0
                        movement = True
                if ptzmode:
                    if mc.Senitivityx > 2 or mc.Senitivityy > 2:
                        mc.Senitivityy = 1
                        mc.Senitivityx = 1
                if not movement:
                    mc.none()
                else:
                    mc.write()
    except Exception as e:
        print(e)

# Function to handle controls based on key pressed
def controls(key_pressed):
    global boundy, boundx, frame, showbounds, persons, selected, detect, alttracking, direct, autocut,mc

    key_actions = {
        "f": lambda: update_bound("x", 25),
        "g": lambda: update_bound("x", -25),
        "c": lambda: update_bound("y", 25),
        "v": lambda: update_bound("y", -25),
        "b": lambda: update_selected(-1),
        "n": lambda: update_selected(1),
        "r": lambda: persons.clear(),
        "e": lambda: toggle("detect"),
        "t": lambda: toggle("alttracking"),
        "p": toggle_direct_autocut
    }

    action = key_actions.get(key_pressed)
    if action:
        action()

def update_bound(axis, change):
    global boundx, boundy, showbounds,frame
    if axis == "x":
        boundx = min(max(boundx + change, 0), frame.shape[1] // 2)
        print(boundx)
    elif axis == "y":
        boundy = min(max(boundy + change, 0), frame.shape[0] // 2)
    showbounds = True

def update_selected(change):
    global selected, persons
    if 0 <= selected + change < len(persons):
        selected += change

def toggle(var_name):
    global detect, alttracking
    globals()[var_name] = not globals()[var_name]
    print(globals()[var_name])

def toggle_direct_autocut():
    global direct, autocut
    direct = not direct
    autocut = direct


# Function to handle preset calls
def callpreset(preset):
    global persons, delay, presetcalled, detect, boundx,debug
    while len(persons) > 0:
        for person in persons:
            persons.remove(person)
    delay = time.time()
    mc.preset(preset)
    presetcalled = True
    print("Preset called!!!")
    detect = False
def directmode():
    global persons,ac,searching,autocut,lastpreset
    print (searching)
    if len(persons)==0:
        searching=True
    else:
        if searching:
            if not debug:
                ac.switchcam(5)
            print("switched to cam 5")
            autocut=True
        searching=False
        lastpreset=0
    if searching:
        if delay+4<time.time():
            if lastpreset>5:
                lastpreset=1
            else:
                lastpreset+=1
            callpreset(lastpreset)
# Function to handle stream deck commands
def stream_deck_command(command):
    global mc, interupt, delay, persons, detect, boundx, boundy, showbounds, autocut, direct,selected
    interupt = True
    print(command)
    delay = time.time()
    if command == "up":
        delay += 10
        mc.u = 1
    elif command == "down":
        delay += 10
        mc.d = 1
    elif command == "left":
        delay += 10
        mc.L = 1
        mc.Senitivityx = 10
    elif command == "right":
        delay += 10
        mc.Senitivityx = 10
        mc.r = 1
    elif command == "stopmove":
        mc.Senitivityx = 2
        mc.none()
        interupt = False
    elif command == "zoomin":
        delay += 10
        mc.zoom = 1
    elif command == "zoomout":
        delay += 10
        mc.zoom = -1
    elif command == "reset":
        while len(persons) > 0:
            for person in persons:
                persons.remove(person)
    elif command == "stop":
        detect = not detect
    if ptzmode:
        mc.Senitivityx = 1
    mc.write()
    if command == "boundsx+":
        showbounds = True
        boundx += 25
        interupt = False
    elif command == "boundsx-":
        showbounds = True
        boundx -= 25
        interupt = False
    if command == "boundsy+":
        showbounds = True
        boundy += 25
        interupt = False
    elif command == "boundsy-":
        showbounds = True
        boundy -= 25
        interupt = False
    elif command == "autocut":
        autocut = not autocut
        print(autocut)
    elif command == "direct":
        direct = not direct
        autocut = direct
    elif command == "toggle":
        detect = not detect
    elif command == "switch":
        selected += 1
        if len(persons) > 0:
                if (selected + 1) > len(persons):
                    selected = 0

# Flask route to handle preset calls
@app.route('/callpreset', methods=['POST'])
def handle_callpreset():
    data = request.json
    preset = data.get("preset")
    move = data.get("move")
    print(move)
    if preset is not None or move is not None:
        if preset is not None:
            threading.Thread(target=callpreset, args=(preset,)).start()
        elif move is not None:
            threading.Thread(target=stream_deck_command, args=(move,)).start()
        return {"status": "success", "message": f"Preset {preset} called"}
    else:
        return {"status": "error", "message": "Missing preset parameter"}, 400

# Function to run the Flask app
def run_flask_app():
    app.run(debug=False, port=5000)
    time.sleep(1)
def detectfaces(face_detection_interval):
    global persons, frame, face_cascade, profile_face, scale_factor2, gray, endthread,faces
    scale_factor = 0.9
    small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    if len(persons) == 0 or face_detection_interval < 10:
        faces = face_cascade.detectMultiScale(small_gray, 1.3, 7, minSize=(100, 100))
        if len(faces) == 0:
            faces = profile_face.detectMultiScale(small_gray, 1.2, 4, minSize=(80, 80))
    if endthread and len(persons) > 0:
        endthread = False
        threading.Thread(target=track).start()
    if face_detection_interval > 20 and len(persons) > 0:
        faces = face_cascade.detectMultiScale(gray, 1.5, 8, minSize=(100, 100))
    if len(faces) == 0:
        if face_detection_interval == 2 or face_detection_interval > 20:
            small_frame = cv2.resize(frame, None, fx=scale_factor2, fy=scale_factor2, interpolation=cv2.INTER_LINEAR)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(small_frame_rgb)
            if boxes is not None:
                faces = [(int(x1 / scale_factor2), int(y1 / scale_factor2), int((x2 - x1) / scale_factor2), int((y2 - y1) / scale_factor2)) for (x1, y1, x2, y2) in boxes]
    return faces

def handlepeaple(faces):
    global persons, frame, box, endthread, face_detection_interval, detect, presetcalled
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 1), (x + w + 1, y + h + 1), (255, 0, 0), 10)
        new_face = (x, y, w, h)
        box.set(new_face)
        matched = False
        for person in persons[:]:
            if matched:
                if person.rect.overlap(box)[0]:
                    persons.remove(person)
            else:
                if person.tracking:
                    matched, dist = person.rect.overlap(box)
                else:
                    matched, dist = person.rect.overlapx(box)
                if matched:
                    person.rect.set(new_face)
                    person.recentPair = True
                    if dist < 10000 or not person.tracking:
                        person.init(frame, new_face)
            if person.rect.xmatch(box):
                matched = True
        if not matched:
            old_len = len(persons)
            new_person = P.Person(tracker_type='KCF')
            new_person.init(frame, new_face)
            persons.append(new_person)
            if old_len == 0 and len(persons) > 0:
                endthread = False
                threading.Thread(target=track).start()
    if face_detection_interval == 2:
        for peaple in persons:
            if not peaple.tracking:
                peaple.confidence -= 1
                if peaple.confidence < 0:
                    persons.remove(peaple)
                    print("Removed")
                print(peaple.confidence)
    if presetcalled and len(persons) == 0 and detect:
                frame_height, frame_width = frame.shape[:2]
                smaller = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_CUBIC)
                val = P.bodysearch(smaller, int(frame_width / 160), frame_height / 3)
                if val is not None:
                    new_person = P.Person(tracker_type='KCF')
                    new_person.init(frame, val)
                    new_person.confidence = 50
                    new_person.tracking = False
                    persons.append(new_person)
                presetcalled = False
def handleGUI():
    global frame, showbounds, boundx, boundy, persons, frame_idx, detect, key_held, direct, autocut, selected, faces,debug
    draw_boxes(frame, [person for person in persons if person.bbox is not None])
    if showbounds:
        cv2.line(frame, (boundx, 0), (boundx, frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (frame.shape[1] - boundx, 0), (frame.shape[1] - boundx, frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (0, int(boundy - (3 * boundy / 4))), (frame.shape[1], int(boundy - (3 * boundy / 4))), (0, 255, 0), 2)
        cv2.line(frame, (0, frame.shape[0] - int(boundy + (3 * boundy / 4))), (frame.shape[1], frame.shape[0] - int(boundy + (3 * boundy / 4))), (0, 255, 0), 2)
        if frame_idx % 20 == 0:
            showbounds = False
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text = "Tracking off"
    if not detect or key_held:
        cv2.putText(frame, text, (20, 100), font, 1, (0, 0, 255), 2)
    if autocut:
        cv2.circle(frame, (20, 20), 25, (255, 0, 0), -1)
    if direct:
        cv2.putText(frame, "DIRECT MODE", (int(frame.shape[1] / 2) - 200, int(frame.shape[0] / 2)), font, 2, (0, 0, 255), 2)
    max_faces = frame.shape[0] // 200
    side_panel_new = np.zeros((frame.shape[0], 100, 3), dtype="uint8")
    try:
        for i, person in enumerate(persons[:max_faces]):
            if person.get_image() is not None:
                person_image = person.get_image()
                if person_image.shape != (200, 100, 3):
                    person_image = cv2.resize(person_image, (100, 200))
                side_panel_new[(i) * 200: (i + 1) * 200, :] = person_image
                if i == selected:
                    cv2.rectangle(side_panel_new, (0, (i) * 200), (100, (i + 1) * 200), (0, 255, 0), 4)
            else:
                persons.remove(person)
                side_panel_new[:, i * 100:(i + 1) * 100] = np.zeros((200, 100, 3), dtype="uint8")
    except Exception as e:
        print(e)
    frame = np.hstack((frame, side_panel_new))
    if frame is None or frame.size == 0:
        raise ValueError("The frame is empty or not valid.")
    if not debug:
        frame = cv2.resize(frame, (2100, 1080), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("GORT", frame)
        cv2.moveWindow("GORT", 0, -50)
    else:
        cv2.imshow("GORT", frame)

# Function to set up argument parser
#main engineer$ python3 cleanPersonFinder.py -camera_IP 192.168.20.206
def get_parser():
    cameraIP = "192.168.20.206"
    port = 1259
    PTZ = False
    debug = False
    communicate = True
    UDP = False
    parser = argparse.ArgumentParser(description="Initial settings for GORT")
    parser.add_argument("-camera_IP", help=f"CAMA, CAM3, CAM5, CAM6 -- default {cameraIP}", default=cameraIP)
    parser.add_argument("-port", type=int, help=f"camera IP port -- default {port}", default=port)
    parser.add_argument("-PTZ",  action='store_true', help=f"PTZ mode (True or False) -- default {PTZ}", default=PTZ)
    parser.add_argument("-UDP", action='store_true', help=f"UDP mode (True or False) -- default {UDP}", default=UDP)
    parser.add_argument("-debug",action='store_true', help=f"Debug mode (True or False) -- default {debug}", default=debug)
    parser.add_argument("-communicate", help=f"Whether Gort should send commands out (True or False) -- default {communicate}", default=communicate)
    return parser

# Function to get parsed arguments
def get_args():
    return get_parser().parse_args()

#################################################################################################################################################################################
# Main function
def main():
    global args, mc, debug, endthread, persons, cap, resized, prev_gray, shared_frame, key_pressed, key_held, detect, presetcalled, alttracking, interupt, delay, listener
    global showbounds,autocut,direct,ac,lastcam,gray,frame,face_cascade,profile_face,upperbody,sizechange,screenwidth,mtcnn,resnet,nn,scale_factor2,selected,searching
    global boundx,boundy,faces,frame_idx,face_detection_interval,box,lastpreset,quitprogram
    
    searching=False
    showbounds = False
    args = get_args()
    debug = args.debug

    mc = M.Mcontrol(args.camera_IP, args.UDP, args.port)
    
    if not debug:
        ac = ATEMController.ATEMControl("192.168.20.177")
    
    alttracking = False
    key_held = False
    endthread = True
    shared_frame = None
    detect = True
    resized = False
    lastpreset=0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    upperbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    sizechange = True
    presetcalled = False
    
    cap = cv2.VideoCapture(0)
    persons = []
    frame_idx = 0
    face_detection_interval = 1
    box = R.Rect()
    selected = 0
    scale_factor2 = 0.25 #used for mtcnn
    
    mtcnn = MTCNN(image_size=160, margin=120, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    if debug:
        listener.start()
    
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = gray
    shared_frame = frame.copy()
    
    screenwidth = 1920
    
    while True:
        resized = False
        ret, frame = cap.read()
        if frame is None or frame.size == 0:
            ret=0
        if presetcalled:
            if not detect and delay + 0.9 < time.time():
                detect = True
                face_detection_interval = 1
        if ret:
            if sizechange:
                try:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                    frame = cv2.convertScaleAbs(frame, alpha=1, beta=0.9)
                except Exception as e:
                    print(e)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shared_frame = frame.copy()
            resized = True

            if frame_idx % face_detection_interval == 0 and detect and not key_held and not alttracking:
                faces=detectfaces(face_detection_interval)
                handlepeaple(faces)

            if direct:
                directmode()
            if len(persons) == 0:
                face_detection_interval = 1
                if autocut:
                    if direct and not debug:
                        lastcam = 4 if lastcam == 6 else 6
                        ac.switchcam(lastcam)
                        autocut = False
                    else:
                        if not debug:
                            ac.switchcam(4)
                        if direct:
                            delay = time.time()
                        autocut = False
            else:
                face_detection_interval = 80
            for peaple in persons:
                if not peaple.tracking:
                    face_detection_interval = 2
                    break
            if len(persons) > 0:
                if (selected + 1) > len(persons):
                    selected = 0
                if not interupt and args.communicate:
                    trackmovement(persons[selected].rect, frame, boundx, boundy, persons[selected].tracking)
                else:
                    if delay + 1 < time.time():
                        interupt = False
            handleGUI()
            frame_idx += 1
            if frame_idx > 1000:
                frame_idx = 0
        else:
            print("not ret")
            cap = cv2.VideoCapture(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quitprogram=True
            break
    endthread = True
    cap.release()
    cv2.destroyAllWindows()
    if debug:
        listener.stop()
    else:
        ac.disconnect()
while not quitprogram:
    try:
        if __name__ == "__main__":
            main()
    except Exception as e:
            exception_str = traceback.format_exc()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open('error_log.txt', 'a') as file:
                file.write(f"{timestamp} - {exception_str}\n")
            print("An error occurred. Check 'error_log.txt' for details. Restarting...")

    
