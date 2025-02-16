import cv2
import numpy as np

# Load the video
input_video_path = 'path_to_your_video.mp4'
output_video_path = 'undistorted_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Load the camera matrix and distortion coefficients
#center_x: Center of X
#centey_y: Center of Y
#focal_length: for horizontaol and vvertical

camera_matrix = np.array([[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, k3, k4])

# Get the width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
    
    # Display the undistorted frame
    cv2.imshow('Undistorted Frame', undistorted_frame)
    
    # Write the undistorted frame to the output video
    out.write(undistorted_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print('Undistorted video saved to', output_video_path)
