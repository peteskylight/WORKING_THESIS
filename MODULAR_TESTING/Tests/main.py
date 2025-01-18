import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QThread, Signal, Slot, QTimer, Qt
import cv2
from PySide6.QtGui import QImage, QPixmap

class VideoProcessor(QThread):
    frame_processed = Signal(object)

    def __init__(self, video_path, resize_frames):
        super().__init__()
        self.video_path = video_path
        self.resize_frames = resize_frames
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.resize_frames:
                frame = cv2.resize(frame, (640, 384))
            self.frame_processed.emit(frame)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analyzer")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel("No video loaded")
        self.video_preview_label = QLabel("Video Preview")
        self.browse_button = QPushButton("Browse Video")
        self.browse_button.clicked.connect(self.browse_video)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.video_preview_label)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.play_pause_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_processor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        self.is_playing = False

    def browse_video(self):
        directory, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if directory:
            self.video_label.setText(f"Loading video: {directory}")
            self.start_video_processing(directory)

    def start_video_processing(self, video_path):
        self.video_processor = VideoProcessor(video_path, resize_frames=True)
        self.video_processor.frame_processed.connect(self.update_frame)
        self.video_processor.start()

    @Slot(object)
    def update_frame(self, frame):
        if self.is_playing:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.video_preview_label.size(), Qt.KeepAspectRatio)
            self.video_preview_label.setPixmap(pixmap)

    def show_next_frame(self):
        if self.video_processor and self.is_playing:
            self.video_processor.frame_processed.connect(self.update_frame)

    def toggle_play_pause(self):
        if self.is_playing:
            self.is_playing = False
            self.play_pause_button.setText("Play")
        else:
            self.is_playing = True
            self.play_pause_button.setText("Pause")
            self.show_next_frame()

    def closeEvent(self, event):
        if self.video_processor:
            self.video_processor.stop()
        event.accept()

if __name__ == "__main__":
    # Check if a QApplication instance already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print("QApplication instance already exists.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
