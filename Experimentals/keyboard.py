import cv2
from pynput import keyboard

import numpy as np


def on_press(key):
    global key_pressed
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            print(f'{key_pressed} pressed')
    except AttributeError:
        pass

def on_release(key):
    global key_pressed
    try:
        if key.char and key.char == key_pressed:
            print(f'{key_pressed} released')
            key_pressed = None  # Reset the key state
    except AttributeError:
        pass

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