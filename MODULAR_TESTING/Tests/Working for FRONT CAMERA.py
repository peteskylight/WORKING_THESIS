import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model (ensure it's a YOLOv8 version with tracking support)
model = YOLO('yolov8m.pt')  
model_conf = 0.4

# Define constants
VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\1ST SURVEY\PROCESSED\Front Cam for Filtering.mp4"  
OUTPUT_VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\OUTPUT DYNAMIC\FOR FINAL CHECKING\FrontDynamic1.mp4"
FPS = 30  
MASK_COLOR = (0, 0, 0)  
MASK_ALPHA = 0.5  

# Define ROI mask function
def create_roi_mask(frame, row_height):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    roi = (0, row_height, width, height)  
    cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)
    return mask

# Initialize video processing
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (frame_width, frame_height))

initial_row_height = int(frame_height * (2 / 12))  # Define mask height

# Tracking persistence dictionary (not needed if using YOLO tracking)
track_history = {}  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi_mask = create_roi_mask(frame, initial_row_height)
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    # Use tracking mode instead of normal detection
    results = model.track(masked_frame, conf=model_conf, persist=True,
                                                  classes=0,
                                                  iou=0.3,
                                                  agnostic_nms=True)

    overlay = frame.copy()
    overlay[roi_mask == 0] = MASK_COLOR  
    output_frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

    # Draw tracking results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1] * len(boxes)

        for box, confidence, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            if int(class_id) != 0:  # Only keep humans
                continue

            x1, y1, x2, y2 = map(int, box)
            y1 = max(y1, initial_row_height)
            y2 = min(y2, frame_height)

            if y1 >= y2:
                continue

            track_id = int(track_id)
            track_history[track_id] = (x1, y1, x2, y2)  

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id} | {confidence:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write and display frame
    out.write(output_frame)
    cv2.imshow('Person Tracking with Masking', output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
