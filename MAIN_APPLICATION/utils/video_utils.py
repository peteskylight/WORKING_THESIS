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

class WhiteFrameGenerator(QThread):
    progress_update = Signal(object)
    return_white_frames = Signal(object)
    
    def __init__(self, main_window, width, height):
        super().__init__()
        self.main_window = main_window
        self.videoWidth = width
        self.videoHeight = height
        self._running = True
        
    def run(self):
        self.main_window.status_import_label.setText("CREATING\nFRAMES")
        self.main_window.white_frames_preview = []
        white_frames = []
        current_frame = 0
        total_frames_length = len(self.main_window.returned_frames_from_browsed_video)
        
        for frame in self.main_window.returned_frames_from_browsed_video:
            white_frame = np.ones((self.videoHeight, self.videoWidth, 3), dtype=np.uint8) * 255 
            
            white_frames.append(white_frame)
            current_frame += 1
            progress = int(current_frame/total_frames_length *100)
            self.progress_update.emit(progress)
            
        self.return_white_frames.emit(white_frames)
    
    def stop(self):
        self._running = False
        self.wait()

class VideoProcessor(QThread):
    frame_processed = Signal(object)
    progress_update = Signal(object)

    def __init__(self, video_path, resize_frames):
        super().__init__()
        self.video_path = video_path
        self.resize_frames = resize_frames
        self._running = True
        self.goSignal = False
        self.videoWidth = None
        self.videoHeight = None

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
            
            del frame
            
        self.frame_processed.emit(frames)
        
        del frames
        cap.release()
    def stop(self):
        self._running = False
        self.wait()