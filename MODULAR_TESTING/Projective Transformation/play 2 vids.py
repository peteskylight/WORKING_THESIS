import cv2

# Load the videos
video_path1 = r"C:\Users\USER\Desktop\WORKING_THESIS\output_original.avi"
video_path2 = r"C:\Users\USER\Desktop\WORKING_THESIS\output_transformed.avi"

cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# Check if videos are opened successfully
if not cap1.isOpened():
    print("Error: Could not open video 1")
if not cap2.isOpened():
    print("Error: Could not open video 2")

# Get video properties
fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
fps = min(fps1, fps2)  # Use the lower FPS to sync the videos

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Scale factor
scale_factor = 1

# Create a window to display the videos
cv2.namedWindow('Two Videos', cv2.WINDOW_NORMAL)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, frame1 = cap1.read()
    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, frame2 = cap2.read()

    # Resize frames to the same size
    frame1 = cv2.resize(frame1, (int(width1 * scale_factor), int(height1 * scale_factor)))
    frame2 = cv2.resize(frame2, (int(width1 * scale_factor), int(height1 * scale_factor)))

    # Ensure frames have the same type
    if frame1.shape != frame2.shape:
        print("Error: Frames do not have the same shape")
        break

    # Concatenate frames horizontally
    combined_frame = cv2.hconcat([frame1, frame2])

    # Display the combined frame
    cv2.imshow('Two Videos', combined_frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release everything
cap1.release()
cap2.release()
cv2.destroyAllWindows()
