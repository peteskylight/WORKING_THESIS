import cv2

cap = cv2.VideoCapture(r"C:\Users\THESIS_WORKSPACE\Desktop\WORKING_THESIS\RESOURCES\Sample Vids\Shorter.mp4")

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))