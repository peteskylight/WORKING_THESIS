import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n-pose.pt')

# Function to process each frame
def process_frame(frame, M, colors, scaled_points):
    height, width, channels = frame.shape
 
    # Perform detection
    results = model(frame, conf=0.3, classes=0, iou=0.5, agnostic_nms=True)

    # Create a white frame
    white_frame = np.ones_like(frame) * 255

    # Draw head centers and border boxes on the white frame
    for idx, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            if box.cls == 0:  # Class ID 0 is for 'person' in COCO dataset
                color = colors[idx % len(colors)]
                # Get the center of the head
                head_center = np.array([[(x1 + x2) / 2, y1]], dtype=np.float32)
                
                # Apply perspective transformation to the head center
                transformed_head_center = cv2.perspectiveTransform(np.array([head_center]), M)[0][0]
                
                # Ensure the transformed head center is within the scaled box
                if (scaled_points[:, 0].min() <= transformed_head_center[0] <= scaled_points[:, 0].max() and
                    scaled_points[:, 1].min() <= transformed_head_center[1] <= scaled_points[:, 1].max()):
                    
                    # Draw the head center on the original frame
                    cv2.circle(frame, (int(head_center[0][0]), int(head_center[0][1])), 5, color.tolist(), -1)
                    
                    # Draw the head center on the white frame
                    cv2.circle(white_frame, (int(transformed_head_center[0]), int(transformed_head_center[1])), 5, color.tolist(), -1)
                    
                    # Draw the border box on the white frame
                    cv2.rectangle(white_frame, (int(transformed_head_center[0] - 10), int(transformed_head_center[1] - 10)),
                                  (int(transformed_head_center[0] + 10), int(transformed_head_center[1] + 10)), color.tolist(), 2)

    # Draw the border box around the scaled frame
    cv2.polylines(white_frame, [np.int32(scaled_points)], isClosed=True, color=(0, 0, 0), thickness=2)

    return frame, white_frame

# Load the video
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\Shorter.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = max(fps, 1)  # Ensure fps is at least 1
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_original = cv2.VideoWriter('output_original.avi', fourcc, fps, (width, height))
out_white_frame = cv2.VideoWriter('output_white_frame.avi', fourcc, fps, (width, height))

# Original points
scale = 0.75  # Scale for box of rows and columns
original_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
# Scaled points
scaled_points = original_points * scale

# Calculate the center offset
center_offset_x = (width - (scaled_points[:, 0].max() - scaled_points[:, 0].min())) / 2
center_offset_y = (height - (scaled_points[:, 1].max() - scaled_points[:, 1].min())) / 2

# Adjust the scaled points to center the box
scaled_points[:, 0] += center_offset_x - scaled_points[:, 0].min()
scaled_points[:, 1] += center_offset_y - scaled_points[:, 1].min()

# Define points for perspective transformation
pts1 = np.float32([[452, 568], [1451, 547], [1915, 534], [8, 1054]])
pts2 = scaled_points

# Apply perspective transformation to the white frame
M = cv2.getPerspectiveTransform(pts1, pts2)

# Generate random colors for bounding boxes
colors = np.random.uniform(0, 255, size=(100, 3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame, white_frame = process_frame(frame, M, colors, scaled_points)

    # Write the frames to the output videos
    out_original.write(processed_frame)
    out_white_frame.write(white_frame)

    # Display the result (optional)
    cv2.imshow('Original', processed_frame)
    cv2.imshow('White Frame', white_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out_original.release()
out_white_frame.release()
cv2.destroyAllWindows()
