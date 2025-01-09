import cv2
import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QThread, Signal, Slot, QMutex
import cv2


class VideoUtils:
    
    def __init__(self) -> None:
        pass

    def save_video(self, output_video_frames, output_video_path, monitorFrames=False):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
        
        for frame in output_video_frames:
            out.write(frame)
            if monitorFrames:
                cv2.imshow("Monitor Frames", frame)
                cv2.waitKey(10)
        out.release()
        
    
    def generate_white_frame(self, height, width):
        white_frame = np.ones((height, width, 3), dtype=np.uint8) * 255 # Create a white frame (all pixel values set to 255)
        return white_frame

class VideoProcessor(QThread):
    frame_processed = Signal(object)

    def __init__(self, video_path, resize_frames):
        super().__init__()
        self.video_path = video_path
        self.resize_frames = resize_frames
        self.frames = []
        self.running = True
        self.mutex = QMutex()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.resize_frames:
                frame = cv2.resize(frame, (640, 384))
            self.mutex.lock()
            self.frames.append(frame)
            self.mutex.unlock()
            self.frame_processed.emit(frame)
        cap.release()
    
    def get_frames(self):
        self.mutex.lock()
        return_frames = self.frames.copy()
        self.mutex.unlock()
        return return_frames

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        self.wait()