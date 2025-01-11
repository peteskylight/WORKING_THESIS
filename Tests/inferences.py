import sys
import time
import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget, QLCDNumber

class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()

        self.video_label = QLabel()
        self.clock_display = QLCDNumber()
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 60)  # Set FPS range from 1 to 60
        self.fps_slider.setValue(30)  # Default FPS

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_slider)
        layout.addWidget(self.clock_display)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(video_path)

        self.video_interval = 1000 // self.fps_slider.value()
        self.clock_interval = 1000  # 1 second interval for clock

        self.video_counter = 0
        self.clock_counter = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_displays)
        self.timer.start(10)  # Base interval of 10 milliseconds

        self.fps_slider.valueChanged.connect(self.change_fps)

    def update_displays(self):
        self.video_counter += self.timer.interval()
        self.clock_counter += self.timer.interval()

        if self.video_counter >= self.video_interval:
            self.update_frame()
            self.video_counter = 0

        if self.clock_counter >= self.clock_interval:
            self.update_clock()
            self.clock_counter = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qimg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends

    def update_clock(self):
        current_time = time.strftime("%H:%M:%S")
        self.clock_display.display(current_time)

    def change_fps(self):
        self.video_interval = 1000 // self.fps_slider.value()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "_main_":
    app = QApplication(sys.argv)
    video_path = "path/to/your/video.mp4"
    player = VideoPlayer(video_path)
    player.resize(800, 600)
    player.show()
    sys.exit(app.exec())