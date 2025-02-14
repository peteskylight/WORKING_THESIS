import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8x.pt')  # Segmentation-enabled model

# Video Paths
VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\1ST SURVEY\PROCESSED\Front Cam for Filtering.mp4" 
OUTPUT_VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\OUTPUT DYNAMIC\FOR FINAL CHECKING\FrontDynamic1.mp4"


# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Define row height thresholds
third_row_y_limit = int(frame_height * (2 / 18))  # Middle of the frame (End of Row 3)

def create_roi_mask(frame):
    """Create a transparent overlay to visualize the filtered area (top part ignored)."""
    mask_overlay = frame.copy()
    cv2.rectangle(mask_overlay, (0, 0), (frame.shape[1], third_row_y_limit), (0, 0, 255), -1)  # Mask from top to middle
    return cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0)  # Blend with transparency

# Track ID Mapping
id_map = {}  # Maps original YOLO IDs to sequential IDs
next_id = 1  # Start counting from 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay mask visualization (Paints top rows in red)
    masked_frame = create_roi_mask(frame)

    # Apply YOLO detection with tracking
    results = model.track(frame, persist=True, conf=0.3, classes = 0, iou = 0.5)

    if results is None:
        continue

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1] * len(boxes)
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Ignore non-human objects (assuming class_id=0 is human)
            if int(class_id) != 0:
                continue  # Ignore chairs, desks, etc.

            # **Filter only first to third row students**
            if y1 < third_row_y_limit:  
                continue  # Ignore students in rows 4, 5, and 6

            # Assign new sequential track ID
            if track_id not in id_map:
                id_map[track_id] = next_id
                next_id += 1  # Increment for the next new person

            sequential_id = id_map[track_id]  # Get the new ID

            # Draw bounding box and track ID
            cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(masked_frame, f"ID {sequential_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write processed frame
    out.write(masked_frame)

    # Display the result
    cv2.imshow('Filtered YOLO Detection with Mask', masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()