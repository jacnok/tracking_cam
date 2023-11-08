import serial
import serial.tools.list_ports

class Mcontrol:
    def __init__(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Device: {port.device}, Name: {port.name}, Description: {port.description}")
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
                self.connected=True
            else:
                
                print("already connected")
            
    def keypressed(self,keycode):
        valid=False
        if keycode== 'w':
            self.u=1
            valid=True
        elif keycode== 's':
            self.d=1
            valid=True
        elif keycode== 'a':
            self.L=1
            valid=True
        elif keycode== 'd':
            self.r=1
            valid=True
        s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
        print(s)
        if self.connected:
            if self.Serial.is_open:
                self.Serial.write(s.encode('utf-8') + b'\n')
        return valid
    def left(self):
        if not self.L ==1:
            self.L=1
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def right(self):
        if not self.r ==1:
            self.r=1
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def up(self):
        if not self.u ==1:
            self.u=1
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def down(self):
        if not self.d ==1:
            self.d=1
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def none(self):
        if self.u==1 or self.d==1 or self.L==1 or self.r==1:
            self.u=0
            self.d=0
            self.L=0
            self.r=0
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def close(self):
        if self.connected:
            if self.Serial.is_open:
                self.Serial.close()