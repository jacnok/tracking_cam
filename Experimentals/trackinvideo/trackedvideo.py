print("Loading...")
import cv2
import numpy as np
import os
from facenet_pytorch import MTCNN
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm

def detect_faces(frame, gray_frame, face_cascade, profile_face, mtcnn, scale_factor2):
    """
    Detect faces in a given frame using Haar Cascades and MTCNN.
    """
    
    # Use Haar Cascade face detection
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 7, minSize=(100, 100))
    
    if len(faces) == 0:
        faces = profile_face.detectMultiScale(gray_frame, 1.2, 4, minSize=(80, 80))
    
    if len(faces) == 0:
        # Use MTCNN for face detection if no Haar Cascade faces were found
        small_frame = cv2.resize(frame, None, fx=scale_factor2, fy=scale_factor2, interpolation=cv2.INTER_AREA)
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(small_frame_rgb)

        if boxes is not None:
            faces = [(int(x1 / scale_factor2), int(y1 / scale_factor2),
                      int((x2 - x1) / scale_factor2), int((y2 - y1) / scale_factor2))
                     for (x1, y1, x2, y2) in boxes]
    else:
        faces = [(int(x ), int(y ),
                  int(w ), int(h))
                 for (x, y, w, h) in faces]
    
    return faces

def combine_audio_video(video_path, audio_path, output_path):
    """
    Combines the given video and audio into a new output file using moviepy.
    """
    print("Combining audio and video...")
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Processing completed. Output saved as {output_path}")

def prompt_user_for_action():
    """
    Prompt the user for an action to perform: whether to continue with visual mode or quit.
    """
    os.system('cls' if os.name == 'nt' else 'clear')  # Clears the console

    global visual_mode ,quit ,speed
    valid_input = False
    while not valid_input:
        print("\n--- Video Processing Options ---")
        print("Enter the video file path or select an option:")
        if visual_mode:
            print("1. Leave Preview mode")
        else:
            print("1. Enter Preview mode")
        print(f"2. Edit speed of program: Set to {speed}")
        print("3. Quit")

        choice = input("Select an option or enter file path : ").strip()

        if choice == "1":
            if visual_mode:
                print("Exiting Preview mode")
            else:
                print("Entered Preview mode")
            visual_mode = not visual_mode
        elif choice == "2":
            print("Edit speed of program:")
            print("The lower the number the slower the program will run but be more accurate.")
            print("The higher the number the faster the program will run but be less accurate.")
            print("range: 5-1000")
            val = input("Enter the speed of the program: ")
            if val.isdigit():
                if int(speed)<5:
                    speed = int(5)
                elif int(speed)>1000:
                    speed = int(1000)
                speed=int(val)
            else:
                print("not valid")
                
        elif choice == "3":
            print("Exiting the program.")
            quit = True
            valid_input = True
        elif os.path.exists(choice):
            valid_input = True
            return choice  # Return valid file path
        else:
            print("Error: Invalid file path. Please try again.")

def main():
    global visual_mode , quit, speed
    quit = False
    visual_mode = False
    speed = 100
    video_input_path = prompt_user_for_action()
    if quit:
        return
    
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Cannot open video capture.")
        return

    # Load face detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    mtcnn = MTCNN(image_size=160, margin=120, keep_all=True)
    scale_factor2 = 0.5  # For MTCNN scaling

    # Get initial frame dimensions and FPS
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Cannot read frame from video capture.")
        return

    original_height, original_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if FPS is not available

    # Resize dimensions for faster processing
    resized_width, resized_height = 320, 240
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    # Video output parameters
    aspect_ratio = 9 / 16
    roi_height = original_height
    roi_width = int(roi_height * aspect_ratio)
    if roi_width > original_width:
        roi_width = original_width
        roi_height = int(roi_width / aspect_ratio)
    half_roi_width = roi_width // 2

    # Video writer for the ROI video
    video_output_path = "output_roi.mp4"
    final_output = "output_with_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (roi_width, roi_height))

    # Progress bar setup
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc='Processing')

    # Initialize variables for tracking and smoothing
    center_x = original_width // 2
    prev_center_x = center_x
    smoothing_factor = 0.1
    tracker = None
    tracking = False
    frame_idx = 0
    face_detection_interval = speed

    # Main loop for video processing
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame for detection and tracking
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        resized_gray = cv2.resize(gray, (resized_width, resized_height))

        if not tracking or frame_idx % face_detection_interval == 0:
            faces = detect_faces(resized_frame, resized_gray, face_cascade, profile_face, mtcnn, scale_factor2)
            if len(faces) > 0 :
                if not tracking:
                    x, y, w, h = faces[0]
                    bbox = (x, y, w, h)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(resized_frame, bbox)
                    tracking = True
                elif len(faces) > 1:
                    for i in range(1, len(faces)):
                        x, y, w, h = faces[i]
                        if abs(center_x-(x+w//2)) < (resized_frame.shape[1]//6):
                            bbox = (x, y, w, h)
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(resized_frame, bbox)
                            tracking = True
                            return
                

        if tracking:
            success, bbox = tracker.update(resized_frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Scale bbox back to original frame size
                x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
                
                new_center_x = x + w // 2
                center_x = int(prev_center_x + smoothing_factor * (new_center_x - prev_center_x))
                prev_center_x = center_x

                # Write the ROI frame to the video
            else:
                tracking = False
        # Adjust ROI bounds
        x1 = max(0, center_x - half_roi_width)
        x2 = min(original_width, center_x + half_roi_width)
        if x2==original_width:
            x1=original_width-roi_width
        roi = original_frame[:, x1:x2]
        roi = cv2.resize(roi, (roi_width, roi_height))
        # Write the ROI frame to the video
        video_writer.write(roi)
        pbar.update(1)

        # Display the resulting frame in visual mode
        if visual_mode:
            cv2.imshow('Face Tracking - ROI', roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    # Cleanup and release resources
    pbar.close()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
   
    # Combine audio and video
    combine_audio_video(video_output_path, video_input_path, final_output)
    
    # Delete the intermediate ROI video
    if os.path.exists(video_output_path):
        os.remove(video_output_path)
        print(f"Deleted intermediate file: {video_output_path}")
if __name__ == "__main__":
    main()
os._exit(0)