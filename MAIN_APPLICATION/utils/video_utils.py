import cv2
import numpy as np

class VideoUtils:
    
    def __init__(self) -> None:
        self.frames = []
        pass

    def read_video(self, video_path, resize_frames):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_frames:
                resized_frame = cv2.resize(frame, (640, 384))
                self.frames.append(resized_frame)
            else:
                self.frames.append(frame)
        cap.release()


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

    def get_frames(self):
        return self.frames