import serial
import serial.tools.list_ports
import socket
import argparse
import time
### 
## Generate Functions
#############################################################################
# Commands from https://f.hubspotusercontent20.net/hubfs/418770/PTZOptics%20Documentation/Misc/PTZOptics%20VISCA%20Commands.pdf

def checkarray(array,array2):
    for i, value in enumerate(array):
        if value != array2[i]:
            return True
    return False
def get_command_map():
    return  {
        "power_on"          : "8101040002FF",
        "power_off"         : "8101040003FF",
        "image_flip_on"     : "8101046602FF",
        "image_flip_off"    : "8101046603FF",
        "camera_flip_off"   : "810104A400FF",
        "camera_flip_H"     : "810104A401FF",
        "camera_flip_V"     : "810104A402FF",
        "camera_flip_VH"    : "810104A403FF",
        "tracking_on"       : "810A115402FF",
        "tracking_off"      : "810A115403FF",
        "execute_preset"    : "8101043F02@FF",
        "pan_up"            : "81010601@#0301FF",
        "pan_down"          : "81010601@#0302FF",
        "pan_left"          : "81010601@#0103FF",
        "pan_right"         : "81010601@#0203FF",
        "pan_up_left"       : "81010601@#0101FF",
        "pan_up_right"      : "81010601@#0201FF",
        "pan_down_left"     : "81010601@#0102FF",
        "pan_down_right"    : "81010601@#0202FF",
        "pan_stop"          : "81010601@#0303FF",
        "zoom_stop"         : "8101040700FF",
        "zoom_tele"         : "8101040702FF",
        "zoom_wide"         : "8101040703FF"
    }


def generate_static_command(command):
    return get_command_map()[command] 

def generate_call_preset_command(preset):
    return get_command_map()['execute_preset'].replace('@', "%0.2X" % preset)

def generate_pan_relative_commands(command, pan_speed, tilt_speed):
    pan_command = get_command_map()[command].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    pan_stop_command = get_command_map()['pan_stop'].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    return pan_command, pan_stop_command
def get_camera_map():
    return  {
        "CAMA" : "192.168.20.200",
        "CAM5A" : "192.168.20.203",
        "CAM5" : "192.168.20.202",
        "CAM6" : "192.168.20.201"
    }

def execute_command(cam_IP, command, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
    s.connect((cam_IP, port))
    data = bytes.fromhex(command)
    s.send(data)
    s.close()

def get_parser():
    # Arg defaults
    camera = 'CAM5A'
    preset = 7
    port = 1259
    command = 'execute_preset'
    pan_speed = 9   # 1-18
    tilt_speed = 7  # 1-14
    pan_duration = 1

    parser = argparse.ArgumentParser(description="Tester calling camera functions")
    parser.add_argument("-camera_name", help=f"CAMA, CAM3, CAM5, CAM6 -- default {camera}", default=camera)
    parser.add_argument("-port", type=int, help=f"camera IP port -- default {port}", default=port)
    parser.add_argument("-command", help=f"Command -- default {command}", default=command)
    parser.add_argument("-preset_num", type=int, help=f"preset number 1-255 -- default {preset}", default=preset)
    parser.add_argument("-pan_speed", type=int, help=f"pan speed 1-18 -- default {pan_speed}", default=pan_speed)
    parser.add_argument("-tilt_speed", type=int, help=f"tilt speed 1-14 -- default {tilt_speed}", default=tilt_speed)
    parser.add_argument("-pan_duration", type=int, help=f"duration for pan commands in seconds -- default {pan_duration}", default=pan_duration)

    return parser

def get_args():
    return get_parser().parse_args()

def print_help():
    get_parser().print_help()

def main():
    args = get_args()

    if args.command not in get_command_map():
        print_help()
        raise Exception(f"Command {args.command} is not found in map")
    if args.camera_name not in get_camera_map():
        print_help()
        raise Exception(f"Camera {args.camera_name} is not found in map")

    verbose_str = f'Executing {args.command} on camera {args.camera_name}'
    if args.command == 'execute_preset':
        if args.preset_num <= 254 and args.preset_num >= 0:
            camera_command = generate_call_preset_command(args.preset_num)
            verbose_str += f' calling preset {args.preset_num}'
        else:
            print_help()
            raise Exception("Preset must be between zero and 254")
        print(verbose_str)
        execute_command(get_camera_map()[args.camera_name], camera_command, port=args.port)
    elif 'pan_' in args.command:
        if args.pan_speed > 18 or args.pan_speed < 1:
            print_help()
            raise Exception("pan_speed must be between 1 and 18")
        if args.tilt_speed > 14 or args.tilt_speed < 1:
            print_help()
            raise Exception("tilt_speed must be between 1 and 14")      
        pan_start, pan_stop = generate_pan_relative_commands(args.command, args.pan_speed, args.tilt_speed)
        verbose_str += f' with pan speed of {args.preset_num} and tilt speed of {args.tilt_speed} for duration of {args.pan_duration}'
        print(verbose_str)
        execute_command(get_camera_map()[args.camera_name], pan_start, port=args.port)
        time.sleep(args.pan_duration)
        execute_command(get_camera_map()[args.camera_name], pan_stop, port=args.port)
    else:
        camera_command = generate_static_command(args.command)
        print(verbose_str)
        execute_command(get_camera_map()[args.camera_name], camera_command, port=args.port)

class Mcontrol:
    def __init__(self,ip="192.168.20.203"):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Device: {port.device}, Name: {port.name}, Description: {port.description}")
        self.u=0
        self.d=0
        self.L=0
        self.r=0
        self.Senitivityx=255
        self.reset=0
        self.zoom=0
        self.connected=False
        self.oldval=[0,0,0,10] #u,d,L,r,z
        self.ip=ip
        if ip=="192.168.20.203":
            self.Senitivityy=5
        else:
            self.Senitivityy=1
        # self.ip="192.168.20.202"
        if len(ports)>0:
            self.Serial = serial.Serial(port.device, 115200)
            self.Serial.close() #it is always open on start for some reason
            if not self.Serial.is_open:
                self.Serial.open()
                print("opened esp")
                self.connected=True
            else:
                
                print("already connected")
            
    def keypressed(self,keycode,keyheld):
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
        elif keycode== 'z':
            self.zoom=1
            valid=True
        elif keycode== 'x':
            self.zoom=-1
            valid=True
        if valid and not keyheld:
            self.Senitivityx=10
            self.write()
        return valid
 
    def none(self):
        if self.oldval[0]==1 or self.oldval[1]==1 or self.oldval[2]==1 or self.oldval[3]==1:
            self.u=0
            self.d=0
            self.L=0
            self.r=0
            self.oldval=[0,0,0,0,10]
            self.Senitivityx=10
            self.zoom=0
            move,stop=generate_pan_relative_commands("pan_up", 8, 2)
            data = bytes.fromhex(stop)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            s.connect((self.ip, 1259))
            s.send(data)
            print("stop moving")
        if self.zoom!=0:
            zoom=get_command_map()["zoom_stop"]
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            s.connect((self.ip, 1259))
            data = bytes.fromhex(zoom) 
            s.send(data)
            s.close()
            self.zoom=0
            print("stop zooming")
    def stopmove(self):
            self.u=0
            self.d=0
            self.L=0
            self.r=0
            self.oldval=[0,0,0,0,10]
            self.Senitivityx=1
            self.zoom=0
            move,stop=generate_pan_relative_commands("pan_up", 8, 2)
            data = bytes.fromhex(stop)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            s.connect((self.ip, 1259))
            s.send(data)
            zoom=get_command_map()["zoom_stop"]
            data = bytes.fromhex(zoom) 
            s.send(data)
            s.close()
            print("Force stop")
    def write(self):
        if self.ip=="192.168.20.202":
            self.Senitivityx=1           
        if self.zoom==-1:
            zoom=get_command_map()["zoom_wide"]
            print("zooming out")
        elif self.zoom==1:
            zoom=get_command_map()["zoom_tele"]
            print("zooming in")
        if self.zoom!=0:
            data = bytes.fromhex(zoom)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            s.connect((self.ip, 1259))
            s.send(data)
            s.close()

        
            
        movement=[self.u,self.d,self.L,self.r,self.Senitivityx]
        if checkarray(movement,self.oldval):
            if self.u==1:
                if self.L==1:
                    movecode,stop=generate_pan_relative_commands("pan_up_left", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                elif self.r==1:
                    movecode,stop=generate_pan_relative_commands("pan_up_right", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                else:
                    movecode,stop=generate_pan_relative_commands("pan_up", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
            elif self.d==1:
                if self.L==1:
                    movecode,stop=generate_pan_relative_commands("pan_down_left", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                elif self.r==1:
                    movecode,stop=generate_pan_relative_commands("pan_down_right", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                else:
                    movecode,stop=generate_pan_relative_commands("pan_down", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
            else:
                if self.oldval[0]==1 or self.oldval[1]==1:
                    s,stop=generate_pan_relative_commands("pan_down", 10, 14)
                    data = bytes.fromhex(stop)
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
                    s.connect((self.ip, 1259))
                    s.send(data)
                    s.close()
                if self.L==1:
                    movecode,stop=generate_pan_relative_commands("pan_left", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                elif self.r==1:
                    movecode,stop=generate_pan_relative_commands("pan_right", self.Senitivityx, self.Senitivityy)
                    data = bytes.fromhex(movecode)
                else:
                    s,stop=generate_pan_relative_commands("pan_right", 10, 14)
                    data = bytes.fromhex(stop)
            
            self.oldval=[self.u,self.d,self.L,self.r,self.Senitivityx]
            if self.oldval[0]==1 or self.oldval[1]==1 or self.oldval[2]==1 or self.oldval[3]==1:#just in case only zoom is changed
                print("movement")
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
                s.connect((self.ip, 1259))
                s.send(data)
                s.close()
    def preset(self,preset):
        if preset<0 or preset>254:
            raise Exception("Preset must be between zero and 254")
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            s.connect((self.ip, 1259))
            camera_command = generate_call_preset_command(preset)
            print(f' calling preset {preset}')
            if self.ip=="192.168.20.203":
                execute_command(get_camera_map()["CAM5A"], camera_command, port=1259)
            elif self.ip=="192.168.20.202":
                execute_command(get_camera_map()["CAM5"], camera_command, port=1259)
    def close(self):
        if self.connected:
            if self.Serial.is_open:
                self.Serial.close()
