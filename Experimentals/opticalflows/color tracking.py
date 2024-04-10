import cv2
import numpy as np

def adjust_features_based_on_color_changes(color_changes, threshold=0.1):
    """
    Adjust the number of features to track based on the intensity of color changes.
    """
    if np.max(color_changes) > threshold:
        return min(100, 25 + int(np.sum(color_changes > threshold) * 2))  # Increase features up to a max of 100
    else:
        return max(5, 25 - int(np.sum(color_changes < threshold)))  # Decrease features down to a min of 5

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

prev_frame = None  # Store the previous frame for color change analysis

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            margin = max(w, h) // 4
            roi_x, roi_y, roi_w, roi_h = max(x - margin // 2, 0), max(y - margin // 2, 0), min(w + margin, gray_frame.shape[1] - x), min(h + margin, gray_frame.shape[0] - y)
            current_roi = gray_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            if prev_frame is not None:
                prev_roi = prev_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                diff = cv2.absdiff(current_roi, prev_roi)

                # Divide the ROI into a grid and analyze color changes in each cell
                grid_size = 10  # Define the size of the grid
                grid_rows, grid_cols = roi_h // grid_size, roi_w // grid_size
                color_changes = np.zeros((grid_rows, grid_cols))
                for i in range(grid_rows):
                    for j in range(grid_cols):
                        cell_y, cell_x = i * grid_size, j * grid_size
                        cell_diff = diff[cell_y:cell_y+grid_size, cell_x:cell_x+grid_size]
                        cell_change = np.mean(cell_diff)
                        color_changes[i, j] = cell_change
                # Use color_changes to adjust bounding box and features
                features_to_track = adjust_features_based_on_color_changes(color_changes)
                # Assuming an existing function to adjust the bounding box based on color_changes
                # adjust_bounding_box(roi_x, roi_y, color_changes)

            # Draw the main face rectangle and the ROI for reference
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 255), 2)

    prev_frame = gray_frame

    cv2.imshow('Face Orientation and Color Change Tracking with Dynamic Features', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
