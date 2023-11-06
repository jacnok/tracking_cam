
import serial
import serial.tools.list_ports
import cv2
import numpy as np
from pynput import keyboard
# Create a black image in memory (e.g., 720x1280 resolution)
ports = serial.tools.list_ports.comports()
    # Print the list of ports
for port in ports:
        print(f"Device: {port.device}, Name: {port.name}, Description: {port.description}")

class Mcontrol:
    def __init__(self):
        self.u=0
        self.d=0
        self.L=0
        self.r=0
        self.Senitivity=255
        self.reset=0
        self.zoom=50
        self.connected=False
        if len(ports)>0:
            self.Serial = serial.Serial(port.device, 115200)
            self.Serial.close() #it is always open on start for some reason
            if not self.Serial.is_open:
                self.Serial.open()
                print("opened esp")
            else:
                
                print("already connected")
            
    def keypressed(self,keycode):
        if keycode== 'w':
            self.u=1
        elif keycode== 's':
            self.d=1
        elif keycode== 'a':
            self.L=1
        elif keycode== 'd':
            self.r=1
        elif keycode== None:
            self.u=0
            self.d=0
            self.L=0
            self.r=0
        s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
        print(s)
        if self.Serial.is_open:
          
          self.Serial.write(s.encode('utf-8') + b'\n')
          
    def close(self):
        if self.Serial.is_open:
            self.Serial.close()
mc=Mcontrol()


def on_press(key):
    global key_pressed
    try:
        if key.char:  # Only consider printable keys
            key_pressed = key.char
            mc.keypressed(key_pressed)
    except AttributeError:
        pass

def on_release(key):
    global key_pressed
    try:
        if key.char and key.char == key_pressed:
           
            key_pressed = None  # Reset the key state
            mc.keypressed(key_pressed)
    except AttributeError:
        pass

# Start listener for key press and release
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()



   
height, width = 480, 480
black_image = np.zeros((height, width, 3), dtype=np.uint8)
# Write a string to the serial port


while True:

    ports = serial.tools.list_ports.comports()
    key=cv2.waitKey(100)
    
    if key == ord('q'):
        break
    cv2.imshow('Black Screen', black_image)
mc.close()
listener.stop()
