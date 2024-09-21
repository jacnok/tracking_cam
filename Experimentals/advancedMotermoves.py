import cv2
from pynput import keyboard
import socket
import numpy as np

ip="192.168.20.206"
port=1259
key_pressed = None
key_held = False
exit=False

def get_command_map():
    return {
        "power_on": "8101040002FF",
        "power_off": "8101040003FF",
        "image_flip_on": "8101046602FF",
        "image_flip_off": "8101046603FF",
        "camera_flip_off": "810104A400FF",
        "camera_flip_H": "810104A401FF",
        "camera_flip_V": "810104A402FF",
        "camera_flip_VH": "810104A403FF",
        "tracking_on": "810A115402FF",
        "tracking_off": "810A115403FF",
        "execute_preset": "8101043F02@FF",
        "pan_up": "81010601@#0301FF",
        "pan_down": "81010601@#0302FF",
        "pan_left": "81010601@#0103FF",
        "pan_right": "81010601@#0203FF",
        "pan_up_left": "81010601@#0101FF",
        "pan_up_right": "81010601@#0201FF",
        "pan_down_left": "81010601@#0102FF",
        "pan_down_right": "81010601@#0202FF",
        "pan_stop": "81010601@#0303FF",
        "zoom_stop": "8101040700FF",
        "zoom_tele": "8101040702FF",
        "zoom_wide": "8101040703FF"
    }


def on_press(key):
    global key_pressed,key_held,exit
    try:
        if key.char:  # Only consider printable keysgg
            key_pressed = key.char
            if key_pressed == 'a' and not key_held:
                print(f'{key_pressed} pressed')
            if key_pressed == 'd' and not key_held:
                move_pan_relative(1, 0, 1)
            key_held = True
            if key_pressed == 'w':#move up
                data,stop=generate_pan_relative_commands("pan_left", 5,5)
                execute_commandTCP(ip,data, port)
            if key_pressed == 's':#move up
                data,stop=generate_pan_relative_commands("pan_right", 5,5)
                execute_commandTCP(ip,data, port)
            if key_pressed == 'q':
                exit=True
    except AttributeError:
        pass

def on_release(key):
    global key_pressed,key_held
    try:
        if key.char and key.char == key_pressed:
            print(f'{key_pressed} released')
            if key_pressed == 'g':
                print(extract_position())
            key_pressed = None  # Reset the key state
            key_held = False
    except AttributeError:
        pass

def extract_position():
        server_address=(ip,port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_address)
        message = bytes.fromhex('81090612FF')
        return_pos = []
        try:
            sock.sendall(message)

            # Look for the response
            amount_received = 0
            amount_expected = 11
            
            while amount_received < amount_expected:
                data = sock.recv(16)
                amount_received += len(data)
                print('received "%s"' % data)
                for c in range(len(data)):
                    v = int(str(data[c]),0)
                    if c in range(2,10):
                        return_pos.append(v)
                    # print(f"{c} : {hex(v)} --> {v}")
        finally:
            sock.close()
        return return_pos

def execute_commandTCP(cam_IP, command, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((cam_IP, port))
    data = bytes.fromhex(command)
    s.sendall(data)
    s.close()

def int_to_hex( value, length):
    """
    Converts an integer to a hexadecimal string of fixed length, padded with leading zeros.
    """
    return f'{value:0{length}X}'

def move_pan_relative( pan_speed, tilt_speed, pan_step=1):
    """
    Adjust the pan position by a single step using a relative move command.
    
    Parameters:
    - pan_speed: Integer (0x01 to 0x18), speed for pan movement.
    - tilt_speed: Integer (0x01 to 0x14), speed for tilt movement.
    - pan_step: Integer, the amount to increase the pan position (default is 1 step).
    """
    
    # Convert parameters to hex strings
    pan_speed_hex = int_to_hex(pan_speed, 2)
    tilt_speed_hex = int_to_hex(tilt_speed, 2)
    pan_position_hex = int_to_hex(pan_step, 8)  # Pan position in hex
    tilt_position_hex = int_to_hex(0, 8)  # No tilt movement (use 0)

    # Construct the VISCA RelativePosition command
    # command = f"81 01 06 03 {pan_speed_hex} {tilt_speed_hex}  f{pan_position_hex[0:2]} {pan_position_hex[2:4]} "f"{tilt_position_hex[0:2]} {tilt_position_hex[2:4]} FF"
    
    command = "81 01 06 03 {} {} {} {} FF".format(pan_speed_hex, tilt_speed_hex,pan_position_hex, tilt_position_hex)
    # command = "81 01 06 03 04 04 00 00 00 03 00 00 00 03 FF"
    # Convert command string to bytes
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    data = bytes.fromhex(command.replace(' ', ''))
    s.sendall(data)
    s.close()

    print(f"Relative Pan Command sent: {command} (Step: {pan_step})")

def generate_pan_relative_commands(command, pan_speed, tilt_speed):
    pan_command = get_command_map()[command].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    pan_stop_command = get_command_map()['pan_stop'].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    return pan_command, pan_stop_command

# Start listener for key press and release
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

height, width = 480, 480
image= np.zeros((height, width, 3), dtype=np.uint8)
# OpenCV window loop
while True:
    # ... your image processing ...

    cv2.imshow('Window', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
        break
    if exit:    # Exit loop if 'q' is pressed
        break
cv2.destroyAllWindows()
listener.stop()
