import cv2
import numpy as np
import rect as R
import People as P
from pynput import keyboard
import MotorController as M
import threading
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request
from PIL import Image

print(cv2.__version__)

boundx = 200
boundy = 125
adjustbounds = True
interupt = False
ptzmode = False
alttracking = False
endthread = True
detect = True
resized = False
sizechange = True
presetcalled = False
shared_frame = None
key_held = False
lines = []
persons = []
frame_idx = 0
selected = 0
scale_factor2 = 0.25
face_detection_interval = 1

mc = M.Mcontrol()
app = Flask(__name__)
cap = cv2.VideoCapture(0)
box = R.Rect()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upperbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

mtcnn = MTCNN(image_size=160, margin=120, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def draw_boxes(frame, people):
    for p in people:
        color = (0, 255, 0) if p.tracking else (0, 0, 255)
        cv2.rectangle(frame, (int(p.rect.x), int(p.rect.y)), (int(p.rect.ex), int(p.rect.ey)), color, 2)


def on_press(key):
    global key_pressed, key_held, interupt, delay
    try:
        if key.char:
            key_pressed = key.char
            if not mc.keypressed(key_pressed, key_held):
                controls(key_pressed)
                key_held = True
            interupt = True
            delay = time.time()
    except AttributeError:
        pass


def on_release(key):
    global key_pressed, key_held
    try:
        if key.char and key.char == key_pressed:
            key_pressed = None
            mc.none()
            key_held = False
    except AttributeError:
        pass


def track():
    global persons, shared_frame, endthread, cap, resized, prev_gray, lines, alttracking
    while not endthread:
        if shared_frame is None:
            time.sleep(0.0001)
        else:
            try:
                if not cap.isOpened():
                    break
                if persons:
                    for person in persons:
                        if not person.update(shared_frame) or alttracking:
                            if alttracking:
                                person.tracking = False
                            lost = person.unpair(prev_gray, gray)
                            if lost:
                                persons.remove(person)
                                print("Removed person")
                else:
                    mc.none()
                    break
                prev_gray = gray.copy()
                shared_frame = None
            except Exception as e:
                print(e)
                break
    endthread = True


def callpreset(preset):
    global persons, delay, presetcalled, detect, boundx
    persons.clear()
    delay = time.time()
    mc.preset(preset)
    presetcalled = True
    boundx = 100 if preset == 1 else 175
    detect = False
    print("Preset called!")


def stream_deck_command(command):
    global mc, interupt, delay, persons, detect, boundx, boundy, showbounds
    interupt = True
    delay = time.time()
    command_map = {
        "up": lambda: setattr(mc, 'u', 1),
        "down": lambda: setattr(mc, 'd', 1),
        "left": lambda: (setattr(mc, 'L', 1), setattr(mc, 'Senitivityx', 10)),
        "right": lambda: (setattr(mc, 'r', 1), setattr(mc, 'Senitivityx', 10)),
        "stopmove": lambda: (setattr(mc, 'Senitivityx', 2), mc.none()),
        "zoomin": lambda: setattr(mc, 'zoom', 1),
        "zoomout": lambda: setattr(mc, 'zoom', -1),
        "reset": lambda: persons.clear(),
        "stop": lambda: setattr(mc, 'detect', not detect),
        "boundsx+": lambda: setattr(mc, 'boundx', boundx + 50),
        "boundsx-": lambda: setattr(mc, 'boundx', boundx - 50),
        "boundsy+": lambda: setattr(mc, 'boundy', boundy + 50),
        "boundsy-": lambda: setattr(mc, 'boundy', boundy - 50),
    }
    command_map.get(command, lambda: None)()
    if ptzmode:
        mc.Senitivityx = 1
    mc.write()


@app.route('/callpreset', methods=['POST'])
def handle_callpreset():
    data = request.json
    preset = data.get("preset")
    move = data.get("move")
    if preset or move:
        threading.Thread(target=callpreset, args=(preset,)).start() if preset else threading.Thread(
            target=stream_deck_command, args=(move,)).start()
        return {"status": "success", "message": f"Preset {preset} called"}
    else:
        return {"status": "error", "message": "Missing preset parameter"}, 400


def run_flask_app():
    app.run(debug=False, port=5000)
    time.sleep(1)


flask_thread = threading.Thread(target=run_flask_app, daemon=True)
flask_thread.start()

ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = gray
shared_frame = frame.copy()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def trackmovment(head, frame, boundx, boundy):
    global detect, key_held, mc
    if detect and not key_held:
        movment = False
        sensitiv = 3
        if head.y < int(boundy - (boundy / 2)):
            cv2.line(frame, (0, int(boundy - (boundy / 2))), (frame.shape[1], int(boundy - (boundy / 2))), (0, 255, 0), 2)
            mc.u = 1
            movment = True
        elif head.ey > frame.shape[0] - int(boundy + (boundy / 2)):
            cv2.line(frame, (0, frame.shape[0] - int(boundy + (boundy / 2))), (frame.shape[1], frame.shape[0] - int(boundy + (boundy / 2))), (0, 255, 0), 2)
            mc.d = 1
            movment = True
        else:
            mc.d = 0
            mc.u = 0

        if head.x < boundx:
            sensitiv = min(max(abs((head.x - boundx) / head.w) * 18, 4), 14)
            cv2.line(frame, (boundx, 0), (boundx, frame.shape[0]), (0, 255, 0), 2)
            mc.Senitivityx = int(sensitiv)
            mc.L = 1
            movment = True
        elif head.ex > frame.shape[1] - boundx:
            sensitiv = min(max(abs((head.ex - (frame.shape[1] - boundx)) / head.w) * 18, 4), 14)
            cv2.line(frame, (frame.shape[1] - boundx, 0), (frame.shape[1] - boundx, frame.shape[0]), (0, 255, 0), 2)
            mc.Senitivityx = int(sensitiv)
            mc.r = 1
            movment = True
        else:
            mc.r = 0
            mc.L = 0

        if head.w < (frame.shape[1] / 8):
            mc.zoom = 1
            movment = True
        elif head.w > (frame.shape[1] / 1.5):
            mc.zoom = -1
            movment = True
        else:
            mc.zoom = 0

        if not movment and not ptzmode:
            microturn = head.cx - (frame.shape[1] / 2)
            if microturn > 15:
                mc.r = 1
                mc.L = 0
                mc.Senitivityx = 1
                movment = True
            elif microturn < -15:
                mc.L = 1
                mc.r = 0
                mc.Senitivityx = 1
                movment = True

        mc.none() if not movment else mc.write()


def controls(key_pressed):
    global boundy, boundx, frame, showbounds, persons, selected, detect, alttracking
    showbounds = False
    key_actions = {
        "f": lambda: setattr(mc, 'boundx', boundx + 25) if boundx < frame.shape[1] // 2 else None,
        "g": lambda: setattr(mc, 'boundx', boundx - 25) if boundx > 0 else None,
        "c": lambda: setattr(mc, 'boundy', boundy + 25) if boundy < frame.shape[0] // 2 else None,
        "v": lambda: setattr(mc, 'boundy', boundy - 25) if boundy > 0 else None,
        "b": lambda: setattr(mc, 'selected', selected - 1) if selected > 0 and len(persons) > 1 else None,
        "n": lambda: setattr(mc, 'selected', selected + 1) if (selected + 1) < len(persons) else None,
        "r": lambda: persons.clear(),
        "e": lambda: setattr(mc, 'detect', not detect),
        "t": lambda: setattr(mc, 'alttracking', not alttracking),
    }
    key_actions.get(key_pressed, lambda: None)()
    showbounds = True


while True:
    resized = False
    ret, frame = cap.read()
    if presetcalled:
        if not detect and delay + 1 < time.time():
            detect = True
            presetcalled = False
            face_detection_interval = 1

    if ret:
        if sizechange:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=0.2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shared_frame = frame.copy()
        resized = True

        if frame_idx % face_detection_interval == 0 and detect and not key_held and not alttracking:
            scale_factor = 0.9
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
            faces = face_cascade.detectMultiScale(small_gray, 1.3, 7, minSize=(100, 100)) or \
                    profile_face.detectMultiScale(small_gray, 1.2, 4, minSize=(80, 80))

            if endthread and persons:
                endthread = False
                threading.Thread(target=track).start()

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (int(x / scale_factor), int(y / scale_factor)),
                              (int((x + w) / scale_factor), int((y + h) / scale_factor)), (0, 0, 255), 1)

            faces = [(int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor))
                     for (x, y, w, h) in faces]

            if not faces:
                small_frame = cv2.resize(frame, None, fx=scale_factor2, fy=scale_factor2, interpolation=cv2.INTER_LINEAR)
                small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(small_frame_rgb)
                if boxes is not None:
                    faces = [(int(x1 / scale_factor2), int(y1 / scale_factor2), int((x2 - x1) / scale_factor2),
                              int((y2 - y1) / scale_factor2)) for (x1, y1, x2, y2) in boxes]

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 1), (x + w + 1, y + h + 1), (255, 0, 0), 10)
                new_face = (x, y, w, h)
                box.set(new_face)
                matched = False
                for person in persons[:]:
                    if matched and person.rect.overlap(box)[0]:
                        persons.remove(person)
                    else:
                        matched, dist = person.rect.overlap(box)
                        if matched:
                            person.rect.set(new_face)
                            person.recentPair = True
                            if dist < 10000 or not person.tracking:
                                person.init(frame, new_face)
                    if person.rect.xmatch(box):
                        matched = True
                if not matched:
                    new_person = P.Person(tracker_type='KCF')
                    new_person.init(frame, new_face)
                    persons.append(new_person)
                    if len(persons) == 1:
                        endthread = False
                        threading.Thread(target=track).start()

        face_detection_interval = 1 if not persons else 80
        for person in persons:
            if not person.tracking:
                face_detection_interval = 2
                break

        draw_boxes(frame, [person for person in persons if person.bbox is not None])
        if showbounds:
            cv2.line(frame, (boundx, 0), (boundx, frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (frame.shape[1] - boundx, 0), (frame.shape[1] - boundx, frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (0, int(boundy - (boundy / 2))), (frame.shape[1], int(boundy - (boundy / 2))), (0, 255, 0), 2)
            cv2.line(frame, (0, frame.shape[0] - int(boundy + (boundy / 2))), (frame.shape[1], frame.shape[0] - int(boundy + (boundy / 2))), (0, 255, 0), 2)
            if frame_idx % 20 == 0:
                showbounds = False

        frame_idx += 1
        if persons:
            if selected + 1 > len(persons):
                selected = 0
            if not interupt:
                trackmovment(persons[selected].rect, frame, boundx, boundy)
            elif delay + 0.7 < time.time():
                interupt = False

        max_faces = frame.shape[1] // 100
        bottom_panel = np.zeros((200, frame.shape[1], 3), dtype="uint8")
        for i, person in enumerate(persons[:max_faces]):
            bottom_panel[:, i * 100:(i + 1) * 100] = person.get_image()
            if i == selected:
                cv2.rectangle(bottom_panel, (i * 100, 0), ((i + 1) * 100, 200), (0, 255, 0), 4)

        frame = np.vstack((frame, bottom_panel))
        side_panel = np.zeros((frame.shape[0], 250, 3), dtype="uint8")
        texts = [
            "Press 'e' key to enable",
            "and disable face",
            "recognition",
            "Press WASD keys to move",
            "Press 'f' or 'g' to",
            "adjust horizontal bounds",
            "Press 'c' or 'v' to",
            "adjust vertical bounds",
            "Press 'b' or 'n' to",
            "switch squares",
            "Press 'r' to reset all",
            "detected squares"
        ]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        for i, text in enumerate(texts):
            cv2.putText(side_panel, text, (10, 20 + (i * 25)), font, .6, (255, 255, 255), 1)

        frame = np.hstack((frame, side_panel))
        if frame_idx > 1000:
            frame_idx = 0

        cv2.imshow("My Face Detection Project", frame)

    else:
        cap = cv2.VideoCapture(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

endthread = True
cap.release()
cv2.destroyAllWindows()
mc.close()
listener.stop()
