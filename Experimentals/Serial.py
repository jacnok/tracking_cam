
import serial
import serial.tools.list_ports
import cv2

ports = serial.tools.list_ports.comports()
esp=False
    # Print the list of ports
for port in ports:
        print(f"Device: {port.device}, Name: {port.name}, Description: {port.description}")
# Create a serial connection
if len(ports)>0:
   
    ser = serial.Serial(port.device, 115200)  # Change 'COM3' to your port and 9600 to your baud rate if different
    # Ensure the serial port is open
    if ser.is_open:
        esp=True
    if not ser.is_open:
      ser.open()
      esp=True

# Write a string to the serial port
while len(ports)>0:
    ports = serial.tools.list_ports.comports()

    if esp:
    # ser.write(b'Hello, World!')  # The 'b' prefix is used to indicate a byte string
        try:
            print(ser.read(100))
        except:
            break
        
    
if esp:
    # Close the serial port
    ser.close()