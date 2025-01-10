import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np

class VideoProcessor(QThread):
    frame_processed = Signal(object)

    def __init__(self, video_path, resize_frames):
        super().__init__()
        self.video_path = video_path
        self.resize_frames = resize_frames
        self._running = True

    def run(self):
        while self._running:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened() and self._running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
                    continue
                if self.resize_frames:
                    frame = cv2.resize(frame, (640, 384))
                self.frame_processed.emit(frame)
            cap.release()

    def stop(self):
        self._running = False
        self.wait()
    #Old Code
    
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

