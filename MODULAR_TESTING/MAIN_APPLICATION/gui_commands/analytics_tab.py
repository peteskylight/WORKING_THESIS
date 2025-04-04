#IMPORTS
import cv2


from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QScreen, QImage, QPixmap

from utils import (VideoProcessor,
                   DrawingBoundingBoxesThread,
                   WhiteFrameGenerator,
                   DrawingKeyPointsThread)
from trackers import (PoseDetection,
                      HumanDetectionThread,
                      PoseDetectionThread)

#Main Importing Tab Class
#I misnamed the class hehe but this is the class for the importing the video and previewing the video

class Analytics:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.human_detect_model = "yolov8n.pt"
        self.human_detect_conf = 0.5
        self.human_pose_model = "yolov8n-pose.pt"
        self.human_pose_conf = 0.5
        
        self.returned_frames = []
        self.humanDetectionResults = []
        self.humanPoseDetectionResults = None
        self.isImportDone = False
        self.videoHeight = None
        self.videoWidth = None
        self.number_of_frames = 0
        
        self.detection = PoseDetection(humanDetectionModel=self.human_detect_model,
                                            humanDetectConf= self.human_detect_conf,
                                            humanPoseModel= self.human_pose_model,
                                            humanPoseConf= self.human_pose_conf
                                            )
    

    def browse_video(self):
        directory, _ = QFileDialog.getOpenFileName(self.main_window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if directory:
            self.main_window.videoDirectory.setText(f"{directory}")
            self.start_video_processing(directory)
            

    def start_video_processing(self, video_path):
        self.main_window.status_import_label.setText("[ GETTING \nFRAMES ]")
        self.video_processor = VideoProcessor(video_path, resize_frames=True)
        self.video_processor.start()
        self.video_processor.frame_processed.connect(self.update_frame_list)
        self.video_processor.progress_update.connect(self.update_import_progress_bar)
        
    def update_frame_list(self, frames):
        self.main_window.returned_frames_from_browsed_video = None
        self.main_window.returned_frames_from_browsed_video = frames
        self.returned_frames = frames
        self.number_of_frames = len(frames)
        self.detectResults(frames)
    
    def detectResults(self,frames):
        self.main_window.status_import_label.setText("[ SCANNING \nHUMANS ]")
        self.main_window.importProgressBar.setValue(0)
        self.human_detection_thread = HumanDetectionThread(
            video_frames = frames,
            main_window=self.main_window,
            humanDetectionModel=self.human_detect_model,
            humanDetectConf=self.human_detect_conf,
            humanPoseModel = self.human_pose_model,
            humanPoseConf=self.human_pose_conf
        )
        self.human_detection_thread.human_track_results.connect(self.update_detection_results)
        self.human_detection_thread.human_detection_progress_update.connect(self.update_import_progress_bar)
        self.human_detection_thread.start()

    def update_detection_results(self, results_list):
        self.humanDetectionResults = results_list
        self.detect_keypoints(results_list)

    def detect_keypoints(self, humanDetectedResults):
        self.main_window.status_import_label.setText("[ DETECTING \n KEYPOINTS ]")
        self.pose_detection_thread = PoseDetectionThread(original_frames=self.returned_frames,
                                                         human_detect_results=humanDetectedResults,
                                                         humanDetectionModel=self.human_detect_model,
                                                         humanDetectConf=self.human_detect_conf,
                                                         humanPoseModel=self.human_pose_model,
                                                         humanPoseConf=self.human_pose_conf
                                                         
                                                         )
        
        
        self.pose_detection_thread.pose_detection_results.connect(self.update_pose_detection_results)
        self.pose_detection_thread.pose_detection_progress_update.connect(self.update_detect_progress_bar)  
        self.pose_detection_thread.start()

    def update_pose_detection_results(self, pose_results):
        self.humanPoseDetectionResults = pose_results
        print(len(pose_results))
        #gENERATE wHITE FRAMES
        self.generate_white_frames()

    def generate_white_frames(self):
        self.video_processor.stop()
        self.main_window.status_import_label.setText("[ CREATING\nFRAMES ]")
        self.white_frame_generator = WhiteFrameGenerator(number_of_frames=self.number_of_frames,
                                                         height=608,
                                                         width=1088)
        self.white_frame_generator.start()
        self.white_frame_generator.progress_update.connect(self.update_import_progress_bar)
        self.white_frame_generator.return_white_frames.connect(self.update_white_frame_list)
        
    
    def update_import_progress_bar(self,value):
        self.main_window.importProgressBar.setValue(value)
        
    def update_detect_progress_bar(self,value):
        self.main_window.importProgressBar.setValue(value)
             
    
    def update_white_frame_list(self, frames):
        self.main_window.white_frames_preview = frames
        self.drawBoundingBoxes()
        
    
    
        
    
    def update_frame(self, frame):
        if frame is not None and self.main_window.is_playing:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.video_preview_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.main_window.video_preview_label.setPixmap(scaled_pixmap)


    def update_white_frame(self, white_frame):
        if white_frame is not None and self.main_window.is_playing:
            
            height, width, channel = white_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(white_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.video_keypoints_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.main_window.video_keypoints_label.setPixmap(scaled_pixmap)
    
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
        
        
    

    def drawBoundingBoxes(self):
        self.main_window.status_import_label.setText("[ DRAWING \nBOXES ]")
        self.draw_bbox = DrawingBoundingBoxesThread(results=self.humanDetectionResults,
                                                    white_frames=self.main_window.white_frames_preview)
        self.draw_bbox.frame_drawn_list.connect(self.update_white_frame_list_then_draw_keypoints)
        self.draw_bbox.progress_updated.connect(self.update_detect_progress_bar)
        self.draw_bbox.start()
    
    def update_white_frame_list_then_draw_keypoints(self, frames):
        self.main_window.white_frames_preview = frames
        self.main_window.status_import_label.setText("[ DRAWING \nKEYPOINTS ]")
        self.draw_keypoints = DrawingKeyPointsThread(white_frames=frames,
                                                      keypoints_list=self.humanPoseDetectionResults,
                                                      human_detections=self.humanDetectionResults)
        
        self.draw_keypoints.frame_drawn_list.connect(self.update_white_frame_last)
        self.draw_keypoints.progress_updated.connect(self.update_detect_progress_bar)
        self.draw_keypoints.start()
    
    def update_white_frame_last(self, frames):
        self.main_window.status_import_label.setText("[ VIDEO IS \nREADY! ]")
        self.main_window.white_frames_preview = frames
        self.main_window.play_pause_button.setEnabled(True)   
    
    
    
    
        

    
        
        
        
    
    def closeEvent(self, event):
        self.video_processor.stop()
        self.white_frame_generator.stop()
        event.accept()