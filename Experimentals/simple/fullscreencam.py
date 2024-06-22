import cv2

def main():
    # Open a video capture device (e.g., webcam)
    cap = cv2.VideoCapture(0)
     # Get the screen resolution
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Check if the capture device is opened successfully
    # cv2.namedWindow("FullScreen", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    if not cap.isOpened():
        print("Error: Couldn't open video capture device")
        return

    while True:
        ret, frame =cap.read()
        if not ret:
            break
        # print(cv2.WND_PROP_FULLSCREEN)
        # print(cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('frame',frame)
        print(screen_width)
        print(screen_height)
        
        frame=cv2.resize(frame, (2100,1080),interpolation=cv2.INTER_CUBIC)
        
        
        cv2.imshow("GORT", frame)
        cv2.moveWindow("GORT", 0, -100)
            # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
