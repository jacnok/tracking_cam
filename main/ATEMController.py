import time
import PyATEMMax
import logging


class ATEMControl:
    def __init__(self, ip ="192.168.20.177"):
        self.switcher = PyATEMMax.ATEMMax()
        self.ip = ip
        self.switcher.connect(ip)

    def switchcam(self,cam):
        # ip = "10.0.0.100"
        counter = 0
        sent=False
        while sent==False:
            if self.switcher.waitForConnection(timeout=5):
                
                self.switcher.setProgramInputVideoSource(0, cam)
                sent=True
            else:
                print("Connection failed")
                counter+=1
                if counter>5:
                    print("Connection failed too many times")
                    sent=True
                    break
                else:
                    print("Trying again")
                    self.switcher.connect(self.ip) 
        print(f"[{time.ctime()}] Switched to camera {cam}")

    def softswitchcam(self,cam):
        # ip = "10.0.0.100"
        counter = 0
        sent=False
        while sent==False:
            if self.switcher.waitForConnection(timeout=5):
                
                self.switcher.setPreviewInputVideoSource(0, cam)
                self.switcher.execAutoME(0)

                sent=True
            else:
                print("Connection failed")
                counter+=1
                if counter>5:
                    print("Connection failed too many times")
                    sent=True
                    break
                else:
                    print("Trying again")
                    self.switcher.connect(self.ip) 
        print(f"[{time.ctime()}] Switched to camera {cam}")
    def findcam(self): #script takes a while to run use threading
        found = False
        for i in range(1, 20):
            for r in range(1, 5):
                if self.switcher.tally.bySource.flags[r].program:
                    found = True
                    break
            if found:
                break
            time.sleep(0.1)
        return r

    def disconnect(self):
        self.switcher.disconnect()
        print("Disconnected from ATEM")

# ip = "10.0.0.100"
# ac = ATEMController(ip)
# for i in range(1,5):
#     ac.switchcam(i)
#     time.sleep(2)
# for i in range(1,5):
#     ac.softswitchcam(i)
#     time.sleep(2)
# print (ac.findcam())
# ac.disconnect()
