import cv2

cap = cv2.VideoCapture("rtsp://admin:Bennett2432@192.168.1.64:554/stream")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Test", frame)

    cv2.waitKey(10)

cap.release()