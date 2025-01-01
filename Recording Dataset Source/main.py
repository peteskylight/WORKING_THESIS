
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from trackers import StudentTracker

from utils import (DrawingUtils,
                   Tools)


class CameraFeed(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flashing Camera Feed")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 700, 500)
        self.label.setAlignment(Qt.AlignCenter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update every 1000 ms (1 second)

        self.cap = cv2.VideoCapture(0)  # Open the default camera

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel
            self.label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraFeed()
    window.show()
    sys.exit(app.exec_())
