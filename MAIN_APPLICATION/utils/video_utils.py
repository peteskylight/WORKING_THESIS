import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np

class VideoUtils:
    
    def __init__(self) -> None:
        pass

    def read_video(self, video_path, resize_frames): #LIST
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_frames:
                resized_frame = cv2.resize(frame, (640, 384))
                frames.append(resized_frame)
            else:
                frames.append(frame)
        cap.release()
        return frames
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

class VideoProcessor(QThread):
    frame_processed = Signal(object)
    progress_update = Signal(object)
    generate_signal = Signal(object)

    def __init__(self, video_path, resize_frames):
        super().__init__()
        self.video_path = video_path
        self.resize_frames = resize_frames
        self._running = True
        self.goSignal = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        frames = []
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.resize_frames:
                frame = cv2.resize(frame, (640, 384))
            frames.append(frame)
            current_frame += 1
            progress = int(current_frame/total_frames *100)
            self.progress_update.emit(progress)
            self.goSignal = False
        self.frame_processed.emit(frames)
        self.generate_signal.emit(not self.goSignal)
        cap.release()
        

    def stop(self):
        self._running = False
        self.wait()