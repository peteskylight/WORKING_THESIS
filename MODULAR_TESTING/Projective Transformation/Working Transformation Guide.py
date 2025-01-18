import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n-pose.pt')

# Load the image
image = cv2.imread(r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Examination Sample Images\1.jpg")
height, width, channels = image.shape

# Scale down the image to 75%
scale_percent = 100
new_width = int(width * scale_percent / 100)
new_height = int(height * scale_percent / 100)
dim = (new_width, new_height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Perform detection
results = model(image)

# Create a white frame
white_frame = np.ones_like(image) * 255

# Original points
scale = 0.75  # Scale for box of rows and columns
original_points = np.float32([[0, new_height], [new_width, new_height], [new_width, 0], [0, 0]])

# Scaled points
scaled_points = original_points * scale

# Define points for perspective transformation
pts1 = np.float32([[4,334], [744,693], [1016,122], [730,74]])
pts2 = scaled_points

# Apply perspective transformation to the white frame
M = cv2.getPerspectiveTransform(pts1, pts2)

# Draw head centers on the white frame
for idx, result in enumerate(results):
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        if box.cls == 0:  # Class ID 0 is for 'person' in COCO dataset
            
            # Generate random colors for bounding boxes
            colors = np.random.uniform(0, 255, size=(len(results[0].boxes), 3))
            color = colors[idx % len(colors)]
            
            # Get the center of the head
            head_center = np.array([[(x1 + x2) / 2, y1]], dtype=np.float32)
            
            # Apply perspective transformation to the head center
            transformed_head_center = cv2.perspectiveTransform(np.array([head_center]), M)[0][0]
            
            # Draw the head center on the white frame
            cv2.circle(image, (int(head_center[0][0]), int(head_center[0][1])), 5, color.tolist(), -1)
            
            # Draw the head center on the white frame
            cv2.circle(white_frame, (int(transformed_head_center[0]), int(transformed_head_center[1])), 5, color.tolist(), -1)

# Combine the original image and the transformed white frame
combined_image = np.hstack((image, white_frame))

# Display the result
cv2.imshow('Result', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
