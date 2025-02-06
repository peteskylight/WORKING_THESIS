import cv2

cap = cv2.VideoCapture(r"C:\Users\Bennett\Documents\WORKING_THESIS\RESOURCES\Examination Sample Videos\Center.mp4")

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))