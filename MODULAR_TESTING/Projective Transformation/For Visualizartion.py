import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load the YOLO model
model = YOLO('yolov8n-pose.pt')

# Open the video file
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\Classroom1.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_original = cv2.VideoWriter('output_original.avi', fourcc, fps, (frame_width, frame_height))
out_transformed = cv2.VideoWriter('output_transformed.avi', fourcc, fps, (frame_width, frame_height))

# Generate random colors for each person
num_colors = 100  # Adjust this number based on the expected number of people
colors = np.random.uniform(0, 255, size=(num_colors, 3))

# Define the number of rows and columns for the table
num_rows = 3
num_cols = 6

# Dictionary to store coordinates for each detected person
person_coords_dict = {}

# Create an empty white frame for accumulation
accumulated_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Scale down the frame to 75%
    scale_percent = 100
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Perform detection with tracking
    results = model.track(frame, persist=True, conf=0.2, iou=0.2, classes=0)

    # Create a white frame
    white_frame = np.ones_like(frame) * 255

    # Original points
    scale = 0.75  # Scale for box of rows and columns
    original_points = np.float32([[0, new_height], [new_width, new_height], [new_width, 0], [0, 0]])

    # Scaled points
    scaled_points = original_points * scale

    # Define points for perspective transformation
    pts1 = np.float32([[0, 272], [639, 281], [439, 35], [175, 43]])
    pts2 = scaled_points

    # Apply perspective transformation to the white frame
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Extract coordinates of detected persons with unique IDs
    person_coords = []
    person_ids = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # Class ID 0 is for 'person' in COCO dataset
                x1, y1, x2, y2 = box.xyxy[0]
                bbox_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)  # Center of the bounding box
                person_coords.append(bbox_center[0])
                person_ids.append(int(box.id))  # Ensure person_id is an integer

    # Apply perspective transformation to the detected persons
    transformed_coords = []
    for coord in person_coords:
        transformed_coord = cv2.perspectiveTransform(np.array([[coord]], dtype=np.float32), M)[0][0]
        transformed_coords.append(transformed_coord)

    # Apply K-means clustering to group detected persons
    if len(transformed_coords) > 0:
        n_clusters = min(len(transformed_coords), num_rows * num_cols)  # Adjust the number of clusters based on the number of detected persons
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(transformed_coords)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Assign fixed row and column coordinates for each detected person based on their cluster center
        cell_width = new_width // num_cols
        cell_height = new_height // num_rows
        assigned_cells = set()
        for idx, center in enumerate(centers):
            row = int(center[1] // cell_height)
            col = int(center[0] // cell_width)
            if (row, col) not in assigned_cells:
                person_coords_dict[idx] = (row, col)
                assigned_cells.add((row, col))
            else:
                # Find the nearest unassigned cell
                min_distance = float('inf')
                nearest_cell = None
                for r in range(num_rows):
                    for c in range(num_cols):
                        if (r, c) not in assigned_cells:
                            distance = np.linalg.norm(np.array([r * cell_height, c * cell_width]) - center)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_cell = (r, c)
                person_coords_dict[idx] = nearest_cell
                assigned_cells.add(nearest_cell)

    # Plot each cluster in a cell of a table
    for idx, coord in enumerate(transformed_coords):
        person_id = person_ids[idx]
        color = colors[person_id % num_colors]
        row, col = person_coords_dict[labels[idx]]
        top_left = (col * cell_width, row * cell_height)
        bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
        cv2.rectangle(white_frame, top_left, bottom_right, (0, 255, 0), -1)

        # Draw circles in the middle of the bounding box
        cv2.circle(frame, (int(person_coords[idx][0]), int(person_coords[idx][1])), 5, (0, 255, 0), -1)

        # Draw text at the top of the bounding box on the frame
        text = f"Row: {row + 1}, Col: {col + 1}"
        cv2.putText(frame, text, (int(person_coords[idx][0]), int(person_coords[idx][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw circles on the accumulated frame
        cv2.circle(accumulated_frame, (int(person_coords[idx][0]), int(person_coords[idx][1])), 5, color, -1)

    # Draw the table grid
    for row in range(num_rows + 1):
        cv2.line(white_frame, (0, row * cell_height), (new_width, row * cell_height), (0, 0, 0), 2)
    for col in range(num_cols + 1):
        cv2.line(white_frame, (col * cell_width, 0), (col * cell_width, new_height), (0, 0, 0), 2)

    # Write the frames to the output videos
    out_original.write(frame)
    out_transformed.write(white_frame)

    # Display the result
    # cv2.imshow('Result', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_original.release()
out_transformed.release()
cv2.destroyAllWindows()

# Save the accumulated frame as an image
cv2.imwrite('accumulated_frame.png', accumulated_frame)
