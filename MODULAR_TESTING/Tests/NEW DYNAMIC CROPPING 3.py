import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (use your trained model if available)
model = YOLO("yolov8n.pt")  # Replace with "your_model.pt" if fine-tuned

# Open video file
video_path = r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\1ST SURVEY\PROCESSED\Front 3 persons.mp4"  # Replace with your actual video path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Y threshold for first 3 rows (adjust experimentally)
y_threshold = int(frame_height * 0.95)  # 55% of the image height

# Output video writer (optional: save results)
output_path = "filtered_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame, conf = 0.5, classes = 0)

    # Loop through detections
    for det in results[0].boxes.data:
        x, y, w, h, conf, cls = det.cpu().numpy()  # Convert to NumPy
        
        # Filter by class (person) and Y-coordinate (only first 3 rows)
        y_bottom = y + h  # Bottom Y of bounding box
        if cls == 0 and y_bottom >= y_threshold:  # Class 0 = Person
            # Draw bounding box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow("Filtered Detection", frame)
    
    # Write frame to output video
    out.write(frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
