
import serial
import serial.tools.list_ports
import cv2
import numpy as np
# Create a black image in memory (e.g., 720x1280 resolution)
ports = serial.tools.list_ports.comports()
esp=False
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
        self.lastkey=-1
        self.connected=False
        # self.Serial
    def keypressed(self,keycode):
        if self.lastkey !=keycode:
            if keycode== ord('w'):
                print(b'w')
            elif keycode== ord('s'):
                print(b's')
            elif keycode== ord('a'):
                print(b'a')
            elif keycode== ord('d'):
                print(b'd')
            elif keycode== -1:
                print(b'hello world')
        # serial.write(b'Hello, World!')
        self.lastkey=keycode


    def connect(self,ser):
        if self.connected==False:
            self.Serial=ser
            # pass
        else:
            print("already connected")

# Create a serial connection



mc=Mcontrol()
if len(ports)>0:
   
    ser = serial.Serial(port.device, 115200)  # Change 'COM3' to your port and 9600 to your baud rate if different
    mc.connect(ser)
    # Ensure the serial port is open
    if ser.is_open:
        esp=True
    if not ser.is_open:
      ser.open()
      esp=True
height, width = 480, 480
black_image = np.zeros((height, width, 3), dtype=np.uint8)
# Write a string to the serial port





while True:

    ports = serial.tools.list_ports.comports()
    key=cv2.waitKey(5)
    mc.keypressed(key)
    if esp:
    # ser.write(b'Hello, World!')  # The 'b' prefix is used to indicate a byte string
        try:
            data=ser.read(100)
        except:
            break
    if key == ord('q'):
        break
    cv2.imshow('Black Screen', black_image)
if esp:
    # Close the serial port
    ser.close()
