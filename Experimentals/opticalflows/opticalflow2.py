import cv2
import numpy as np

posx= 640/2
posy=480/2
# Function to draw arrows representing optical flow
def draw_arrows(frame, flow, step=20):
    global posx,posy
    size=100
    ax=np.mean(flow[..., 0])
    ay=np.mean(flow[..., 1])
    h, w = frame.shape[:2]
    sx=posx-size
    ex=posx+size
    sy=posy-size
    ey=posy+size
    if(sx<0):
        sx=0
    if ex>w:
        ex=w
    if(sy<0):
        sy=0
    if ey>h:
        ey=h
    y, x = np.mgrid[sy:ey:step, sx:ex:step].reshape(2, -1).astype(int)
    # y, x = np.mgrid[step//2:h:step, step//2:w:(size)].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Draw arrows
    lines = np.vstack([x, y, x+(fx-ax)*np.absolute(fx), y+(fy-ay)*np.absolute(fy)]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    avgpx=0
    avgpy=0
    amount=0
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
        if (np.sqrt((x1-x2)**2 +(y1-y2)**2)>9):
            avgpx+=x1
            avgpy+=y1
            amount+=1
            # cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
            cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    if(amount):
        avgpx=(avgpx/amount)
        avgpy=(avgpy/amount)
        cv2.circle(frame, (int(avgpx),int(avgpy)), 5, (255, 0, 0), -1)
        posx=(avgpx+posx)/2
        posy=(avgpy+posy)/2
    cv2.circle(frame, (int(posx),int(posy)), 5, (0, 0, 255), -1)

# Capture video from a file or camera
cap = cv2.VideoCapture(0)  # Replace with 0 for webcam

ret, prev_frame = cap.read()

prev_frame = cv2.resize(prev_frame, (640,480), interpolation=cv2.INTER_CUBIC)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 4, 1, 1, .3, 0)
    draw_arrows(frame, flow)

    cv2.imshow('Optical Flow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
