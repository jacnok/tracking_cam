import socket

# Generate Functions
#############################################################################
# Commands from https://f.hubspotusercontent20.net/hubfs/418770/PTZOptics%20Documentation/Misc/PTZOptics%20VISCA%20Commands.pdf

def checkarray(array, array2):
    for i, value in enumerate(array):
        if value != array2[i]:
            return True
    return False

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

def generate_static_command(command):
    return get_command_map()[command]

def generate_call_preset_command(preset):
    return get_command_map()['execute_preset'].replace('@', "%0.2X" % preset)

def generate_pan_relative_commands(command, pan_speed, tilt_speed):
    pan_command = get_command_map()[command].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    pan_stop_command = get_command_map()['pan_stop'].replace('@', "%0.2X" % pan_speed).replace('#', "%0.2X" % tilt_speed)
    return pan_command, pan_stop_command


def execute_commandTCP(cam_IP, command, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((cam_IP, port))
    data = bytes.fromhex(command)
    s.sendall(data)
    s.close()

def execute_commandUDP(cam_IP, command, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use SOCK_DGRAM for UDP
    data = bytes.fromhex(command)
    s.sendto(data, (cam_IP, port))  # Use sendto for UDP
    s.close()

class Mcontrol:
    def __init__(self,ip,UDP, port):
        self.u = 0
        self.d = 0
        self.L = 0
        self.r = 0
        self.Senitivityx = 10
        self.Senitivityy = 10
        self.zoom = 0
        self.oldval = [0, 0, 0, 10]  # u, d, L, r, z
        self.ip = ip
        self.tcp=not UDP
        self.port=port

    def keypressed(self, keycode, keyheld):
        valid = False
        if keycode == 'w':
            self.u = 1
            valid = True
        elif keycode == 's':
            self.d = 1
            valid = True
        elif keycode == 'a':
            self.L = 1
            valid = True
        elif keycode == 'd':
            self.r = 1
            valid = True
        elif keycode == 'z':
            self.zoom = 1
            valid = True
        elif keycode == 'x':
            self.zoom = -1
            valid = True
        if valid and not keyheld:
            self.Senitivityx = 14
            self.Senitivityy = 14
            self.write()
        return valid

    def none(self):
        if self.oldval[0] == 1 or self.oldval[1] == 1 or self.oldval[2] == 1 or self.oldval[3] == 1:
            self.u = 0
            self.d = 0
            self.L = 0
            self.r = 0
            self.oldval = [0, 0, 0, 0, 10]
            self.Senitivityx = 10
            self.zoom = 0
            move, stop = generate_pan_relative_commands("pan_up", 8, 2)
            if self.tcp:
                execute_commandTCP(self.ip, stop, self.port)
            else:
                execute_commandUDP(self.ip, stop, self.port)
            print("stop moving")
        if self.zoom != 0:
            zoom = get_command_map()["zoom_stop"]
            if self.tcp:
                execute_commandTCP(self.ip, zoom, self.port)
            else:
                execute_commandUDP(self.ip, zoom, self.port)
            self.zoom = 0
            print("stop zooming")

    def stopmove(self):
        self.u = 0
        self.d = 0
        self.L = 0
        self.r = 0
        self.oldval = [0, 0, 0, 0, 10]
        self.Senitivityx = 1
        self.zoom = 0
        move, stop = generate_pan_relative_commands("pan_up", 8, 2)
        if self.tcp:
            execute_commandTCP(self.ip, stop, self.port)
        else:
            execute_commandUDP(self.ip, stop, self.port)
        zoom = get_command_map()["zoom_stop"]
        if self.tcp:
            execute_commandTCP(self.ip, zoom, self.port)
        else:
            execute_commandUDP(self.ip, zoom, self.port)
        print("Force stop")

    def write(self):
        if self.zoom == -1:
            zoom = get_command_map()["zoom_wide"]
            print("zooming out")
        elif self.zoom == 1:
            zoom = get_command_map()["zoom_tele"]
            print("zooming in")
        if self.zoom != 0:
            if self.tcp:
                execute_commandTCP(self.ip, zoom, self.port)
            else:
                execute_commandUDP(self.ip, zoom, self.port)
        movement = [self.u, self.d, self.L, self.r, self.Senitivityx]
        if checkarray(movement, self.oldval):
            if self.u == 1:
                if self.L == 1:
                    movecode, stop = generate_pan_relative_commands("pan_up_left", self.Senitivityx, self.Senitivityy)
                elif self.r == 1:
                    movecode, stop = generate_pan_relative_commands("pan_up_right", self.Senitivityx, self.Senitivityy)
                else:
                    movecode, stop = generate_pan_relative_commands("pan_up", self.Senitivityx, self.Senitivityy)
            elif self.d == 1:
                if self.L == 1:
                    movecode, stop = generate_pan_relative_commands("pan_down_left", self.Senitivityx, self.Senitivityy)
                elif self.r == 1:
                    movecode, stop = generate_pan_relative_commands("pan_down_right", self.Senitivityx, self.Senitivityy)
                else:
                    movecode, stop = generate_pan_relative_commands("pan_down", self.Senitivityx, self.Senitivityy)
            else:
                if self.oldval[0] == 1 or self.oldval[1] == 1:
                    s, stop = generate_pan_relative_commands("pan_down", 10, 14)
                    if self.tcp:
                        execute_commandTCP(self.ip, stop, self.port)
                    else:
                        execute_commandUDP(self.ip, stop, self.port)
                if self.L == 1:
                    movecode, stop = generate_pan_relative_commands("pan_left", self.Senitivityx, self.Senitivityy) 
                elif self.r == 1:
                    movecode, stop = generate_pan_relative_commands("pan_right", self.Senitivityx, self.Senitivityy)
                else:
                    s, stop = generate_pan_relative_commands("pan_right", 10, 14)
                    movecode=stop

            self.oldval = [self.u, self.d, self.L, self.r, self.Senitivityx]
            if self.oldval[0] == 1 or self.oldval[1] == 1 or self.oldval[2] == 1 or self.oldval[3] == 1:
                if self.tcp:
                    execute_commandTCP(self.ip, movecode, self.port)
                else:
                    execute_commandUDP(self.ip, movecode, self.port)

    def preset(self, preset):
        if preset < 0 or preset > 254:
            raise Exception("Preset must be between zero and 254")
        else:
            camera_command = generate_call_preset_command(preset)
            print(f' calling preset {preset}')
            if self.tcp:
                execute_commandTCP(self.ip, camera_command, self.port)
            else:
                execute_commandUDP(self.ip, camera_command, self.port)
    def extract_position(self):
        server_address=("192.168.20.206",1259)
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
