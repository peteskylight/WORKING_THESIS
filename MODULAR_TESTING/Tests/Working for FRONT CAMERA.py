import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Use any YOLOv8 model variant as per your requirement

# Define constants
VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\Examination Sample Videos\Shorter.mp4"  # Replace with your video path
OUTPUT_VIDEO_PATH = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\OUTPUT DYNAMIC\Dynamic1.mp4"
FPS = 30  # Assuming 30 FPS, adjust based on your video
MASK_COLOR = (0, 0, 0)  # Black color for masking

# Define ROI for the first 3 columns (masking other areas)
def create_roi_mask(frame, row_height):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Assume seating arrangement: 6 rows × 7 columns
    # Mask everything except the bottom to middle rows
    roi = (0, row_height, width, height)  # Bottom to middle

    # Create a filled rectangle for ROI
    cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)

    return mask

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (frame_width, frame_height))

# Define initial row height (bottom to middle)
initial_row_height = int(frame_height * (4 / 6))  # Bottom 4 rows (adjust as needed)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create initial ROI mask
    roi_mask = create_roi_mask(frame, initial_row_height)

    # Combine the mask with the original image for detection
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    # Apply YOLO model on the masked frame
    results = model(masked_frame)

    # Create a copy of the frame for drawing
    output_frame = frame.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Clip the bounding box to the ROI
            y1 = max(y1, initial_row_height)  # Clip the top of the bounding box to the ROI boundary
            y2 = min(y2, frame_height)  # Clip the bottom of the bounding box to the frame boundary

            # Check if the adjusted bounding box is valid
            if y1 >= y2:  # Invalid bounding box (e.g., fully outside the ROI after adjustment)
                continue

            # Draw bounding boxes on the output frame
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[int(class_id)]} {confidence:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to output
    out.write(output_frame)

    # Display the frame
    cv2.imshow('YOLO Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()