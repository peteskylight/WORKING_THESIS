import cv2

# Replace with your RTSP URL
FRONT_CAMERA_RTSP_URL = "rtsp://admin:Bennett2432@192.168.1.64:554/stream"

# Open the RTSP stream
cap = cv2.VideoCapture(FRONT_CAMERA_RTSP_URL)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow("CCTV Live Feed", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()