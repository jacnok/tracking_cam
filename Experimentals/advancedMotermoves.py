import cv2
from pynput import keyboard
import socket
import numpy as np

ip="192.168.20.206"
port=1259
key_pressed = None
key_held = False
def on_press(key):
    global key_pressed,key_held
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            if key_pressed == 'a' and not key_held:
                print(f'{key_pressed} pressed')
            if key_pressed == 'd' and not key_held:
                 move_pan_relative(1, 1, 1)
            key_held = True
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
                # print('received "%s"' % data)
                for c in range(len(data)):
                    v = int(str(data[c]),0)
                    if c in range(2,10):
                        return_pos.append(v)
                    # print(f"{c} : {hex(v)} --> {v}")
        finally:
            sock.close()
        return return_pos



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
    pan_position_hex = int_to_hex(pan_step, 4)  # Pan position in hex
    tilt_position_hex = int_to_hex(0, 4)  # No tilt movement (use 0)

    # Construct the VISCA RelativePosition command
    command = f"81 01 06 03 {pan_speed_hex} {tilt_speed_hex} " \
                f"{pan_position_hex[0:2]} {pan_position_hex[2:4]} " \
                f"{tilt_position_hex[0:2]} {tilt_position_hex[2:4]} FF"

    # Convert command string to bytes
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    data = bytes.fromhex(command.replace(' ', ''))
    s.sendall(data)
    s.close()

    print(f"Relative Pan Command sent: {command} (Step: {pan_step})")



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

cv2.destroyAllWindows()
listener.stop()
