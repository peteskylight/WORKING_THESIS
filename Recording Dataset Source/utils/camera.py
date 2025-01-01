import cv2
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap

class CameraFeed:
    def __init__(self, label):
        self.label = label
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self, index):
        self.cap = cv2.VideoCapture(index)  
        self.timer.start(10) 

    def update_frame(self): #GET THE FRAME HERE
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel
            self.label.setPixmap(QPixmap.fromImage(q_img))
            
        return frame

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
