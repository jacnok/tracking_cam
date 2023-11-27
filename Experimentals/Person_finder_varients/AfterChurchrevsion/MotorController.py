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
        self.oldval=[0,0,0,0] #u,d,L,r
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
        self.write()
        return valid
 
    def none(self):
        if self.oldval[0]==1 or self.oldval[1]==1 or self.oldval[2]==1 or self.oldval[3]==1:
            self.u=0
            self.d=0
            self.L=0
            self.r=0
            self.oldval=[0,0,0,0]
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def write(self):
        if self.oldval[0] is not self.u or self.oldval[2] is not self.L or self.oldval[1] is not self.d or self.oldval[3] is not self.r:
            self.oldval=[self.u,self.d,self.L,self.r]
            s=str(self.u)+','+str(self.d)+','+str(self.L)+','+str(self.r)+','+str(self.zoom)+','+str(self.Senitivity)+','+str(self.reset)+';'
            print(s)
            if self.connected:
                if self.Serial.is_open:
                    self.Serial.write(s.encode('utf-8') + b'\n')
    def close(self):
        if self.connected:
            if self.Serial.is_open:
                self.Serial.close()