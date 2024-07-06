import time
import PyATEMMax

switcher = PyATEMMax.ATEMMax()
count = 0


def switchcam(cam):
    ip = "192.168.20.177"
    switcher.connect(ip,5,0)
    if switcher.waitForConnection(infinite=False,waitForFullHandshake=False):
        switcher.setPreviewInputVideoSource(0, cam)
    else:
        print(f"[{time.ctime()}] No ATEM switcher found at {ip}")
    switcher.disconnect()
    print(f"[{time.ctime()}] Switched to camera {cam}")