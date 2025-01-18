import cv2
import numpy as np
from ultralytics import YOLO
# Load the YOLO model
model = YOLO('yolov8n.pt')
# Load the image
image = cv2.imread(r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Examination Sample Images\1.jpg")
height, width, channels = image.shape
# Scale down the image to 75%
scale_percent = 75
new_width = int(width * scale_percent / 100)
new_height = int(height * scale_percent / 100)
dim = (new_width, new_height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# Perform detection
results = model(image)
# Create a white frame
white_frame = np.ones_like(image) * 255
# Define points for affine transformation
pts1 = np.float32([[0, 0], [new_width, 0], [0, new_height]])
pts2 = np.float32([[0, 0], [new_width, 0], [int(0.2 * new_width), new_height]])
# Apply affine transformation to the white frame
M = cv2.getAffineTransform(pts1, pts2)
# Draw bounding boxes around detected persons and apply affine transformation to coordinates
for idx, result in enumerate(results):
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        if box.cls == 0:  # Class ID 0 is for 'person' in COCO dataset
            
            
            # Generate random colors for bounding boxes
            colors = np.random.uniform(0, 255, size=(len(results[0].boxes), 3))
            color = colors[idx % len(colors)]
            # Draw bounding box on the original image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color.tolist(), 2)
            
            # Apply affine transformation to the coordinates
            pts = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            transformed_pts = cv2.transform(np.array([pts]), M)[0]
            tx1, ty1 = transformed_pts[0]
            tx2, ty2 = transformed_pts[3]
            
            # Draw bounding box on the white frame with transformed coordinates
            cv2.rectangle(white_frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color.tolist(), 2)
# Combine the original image and the transformed white frame
combined_image = np.hstack((image, white_frame))
# Display the result
cv2.imshow('Result', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()