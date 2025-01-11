#IMPORTS
import cv2


from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QScreen, QImage, QPixmap

from utils import VideoProcessor, DrawingBoundingBoxesThread
from trackers import PoseDetection, PoseDetectionThread

#Main Analytics Tab Class

class Analytics:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.human_detect_model = "yolov8n.pt"
        self.human_detect_conf = 0.4
        self.human_pose_model = "yolov8n-pose.pt"
        self.human_pose_conf = 0.4
        
        self.detectionResults = None
        self.isImportDone = False
        self.videoHeight = None
        self.videoWidth = None
        
        self.detection = PoseDetection(humanDetectionModel='yolov8n.pt',
                                            humanDetectConf=0.4,
                                            humanPoseModel='yolov8n-pose.pt',
                                            humanPoseConf=0.4
                                            )
    

    def browse_video(self):
        directory, _ = QFileDialog.getOpenFileName(self.main_window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if directory:
            self.main_window.videoDirectory.setText(f"{directory}")
            self.start_video_processing(directory)
            
            # Get the height and width of the video
            cap = cv2.VideoCapture(directory)
            if cap.isOpened():
                self.videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.main_window.videoWidth = self.videoWidth
                self.main_window.videoHeight = self.videoHeight
                cap.release()

            
            

    def start_video_processing(self, video_path):
        self.video_processor = VideoProcessor(video_path, resize_frames=True)
        self.video_processor.start()
        self.video_processor.frame_processed.connect(self.update_frame_list)
        self.video_processor.progress_update.connect(self.update_import_progress_bar)
        self.video_processor.generate_signal.connect(self.goSignal)
        
        # self.detectResults()
        
    def goSignal(self, signal):
        if signal:
            self.main_window.white_frames_preview = []
            for frame in self.main_window.returned_frames_from_browsed_video:
                white_frame = self.main_window.video_utils.generate_white_frame(height = self.videoHeight,
                                                                                width = self.videoWidth)
                self.main_window.white_frames_preview.append(white_frame)
        

    def update_import_progress_bar(self,value):
        self.main_window.importProgressBar.setValue(value)
        if value == 100:
            self.main_window.play_pause_button.setEnabled(True)
            self.isImportDone= True
        else:
            self.main_window.play_pause_button.setEnabled(False)
        
    def update_detect_progress_bar(self,value):
        self.main_window.importProgressBar.setValue(value)
        if value == 100:
            self.main_window.play_pause_button.setEnabled(True)        
    
    
    def update_frame_list(self, frames):
        self.main_window.returned_frames_from_browsed_video = None
        self.main_window.returned_frames_from_browsed_video = frames
    
    def update_frame(self, frame):
        if frame is not None and self.main_window.is_playing:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.main_window.video_preview_label.size(), Qt.KeepAspectRatio)
            self.main_window.video_preview_label.setPixmap(pixmap)

    def update_white_frame(self, white_frame):
        if white_frame is not None and self.main_window.is_playing:
            
            white_frame = cv2.cvtColor(white_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = white_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(white_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.main_window.video_keypoints_label.size(), Qt.KeepAspectRatio)
            self.main_window.video_keypoints_label.setPixmap(pixmap)
    
    def show_next_frame(self):
        if self.video_processor and self.main_window.is_playing:
            self.video_processor.frame_processed.connect(self.main_window.update_frame)

    def update_frame_processing(self):
        fps = self.main_window.fps_loading_rate_slider.value()
        self.main_window.fps_flider_value = fps

    def toggle_play_pause(self):
        if self.main_window.is_playing:
            self.main_window.is_playing = False
            self.main_window.play_pause_button.setText("PLAY")
        else:
            self.main_window.is_playing = True
            self.main_window.play_pause_button.setText("PAUSE")
        
        
    def detectResults(self):
        video_path = self.main_window.videoDirectory.text()
        if video_path is not None:
            self.pose_detection_thread = PoseDetectionThread(
                video_path=video_path,
                humanDetectionModel=self.human_detect_model,
                humanDetectConf=0.5,
                humanPoseModel = self.human_pose_model,
                humanPoseConf=0.5
            )
            self.pose_detection_thread.processed_results.connect(self.update_detection_results)
            self.pose_detection_thread.human_detection_progress_update.connect(self.update_import_progress_bar)
            self.pose_detection_thread.start()

    def drawBoundingBoxes(self):
        self.draw_bbox = DrawingBoundingBoxesThread(results=self.detectionResults,
                                                    white_frames=self.main_window.white_frames_preview)
    
    def update_detection_results(self, results_list):
        self.detectionResults = results_list
    
    def closeEvent(self, event):
        self.video_processor.stop()
        event.accept()