import time
import PyATEMMax
import logging

switcher = PyATEMMax.ATEMMax()
count = 0

# Define the transition rate (in frames)
TRANSITION_RATE = 100  # Adjust this value as needed

# currently only works as a janky transition bc
# switcher.execAutoME(0) is being ignored bc of a switcher disconnect
# so the sloppy code below is a workaround

def switchcam(cam):
    ip = "192.168.20.177"
    switcher.connect(ip,5,0)
    if switcher.waitForConnection(infinite=False,waitForFullHandshake=False):

        # Use to set the program input source via hard cuts.
        # switcher.setProgramInputVideoSource(0, cam)

        # added logging so we can see what's going on
        switcher.setLogLevel(logging.DEBUG)
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)


        switcher.setPreviewInputVideoSource(0, cam)
        switcher.setTransitionStyle(0, PyATEMMax.ATEMTransitionStyles.mix)

         # Set the transition rate
        switcher.setTransitionMixRate(0, TRANSITION_RATE)
        print(f"Transition rate set to {TRANSITION_RATE} frames")       
        
        # never works bc of a switcher disconnect???
        switcher.execAutoME(0)

        # this is a janky workaround
        for i in range(1, 9999):
            switcher.setTransitionPosition(PyATEMMax.ATEMMixEffects.mixEffect1, i)

  
    else:
        print(f"[{time.ctime()}] No ATEM switcher found at {ip}")

    switcher.disconnect()
    print(f"[{time.ctime()}] Switched to camera {cam}")
