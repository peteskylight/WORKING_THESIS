import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Use any YOLOv8 model variant as per your requirement

# Define constants
VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\Examination Sample Videos\Shorter.mp4"  # Replace with your video path
OUTPUT_VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\OUTPUT DYNAMIC\1.mp4"
FPS = 30  # Assuming 30 FPS, adjust based on your video
MASK_COLOR = (0, 0, 0)  # Black color for masking

# Define ROI for the first 3 columns (masking other areas)
def create_roi_mask(frame):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Assume seating arrangement: 6 rows × 7 columns
    # Mask everything except the first 3 columns
    col_width = width
    row_height = height // 6
    roi = (0, row_height * 4, col_width, height)  # Bottom to middle

    # Create a filled rectangle for ROI
    cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)

    cv2.imshow('YOLO Detection', mask)
    return mask

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create ROI mask
    roi_mask = create_roi_mask(frame)
    
    # Apply YOLO model on frame
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            # Check if the center of the box lies within the ROI
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            if roi_mask[center_y, center_x] > 0:
                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(class_id)]} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to output
    out.write(frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
