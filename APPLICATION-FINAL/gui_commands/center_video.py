import cv2

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QScreen, QImage, QPixmap

from utils import (VideoProcessor,
                   WhiteFrameGenerator,
                   DrawingBoundingBoxesThread,
                   DrawingKeyPointsThread)
from trackers import (PoseDetection,
                      HumanDetectionThread,
                      PoseDetectionThread,
                      ActionDetectionThread)

class CenterVideo:
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
        

    #When browse for center video is clicked
    def browse_video(self):
        self.main_window.play_pause_button_video_center.setText("PLAY")
        self.main_window.play_pause_button_video_center.setEnabled(False)
        self.human_detect_conf = (int(self.main_window.center_video_human_conf_slider.value())/100)
        self.human_pose_conf = (int(self.main_window.center_video_keypoint_conf_slider.value())/100)
        directory, _ = QFileDialog.getOpenFileName(self.main_window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if directory:
            self.main_window.videoDirectory_center.setText(f"{directory}")
            self.start_video_processing(directory)
    
    def start_video_processing(self, video_path):
        self.main_window.status_label_center.setText("[ GETTING FRAMES ]")
        self.video_processor = VideoProcessor(video_path, resize_frames=False)
        self.video_processor.start()
        self.video_processor.frame_processed.connect(self.update_frame_list)
        self.video_processor.progress_update.connect(self.update_progress_bar)
     
    def update_frame_list(self, frames):
        self.main_window.returned_frames_from_browsed_center_video = None
        self.main_window.returned_frames_from_browsed_center_video = frames
        self.returned_frames = frames
        self.number_of_frames = len(frames)
        self.detectResults(frames)

    def update_video_frames_list_only(self, frames):
        self.main_window.returned_frames_from_browsed_center_video = frames
        self.returned_frames = frames
    
    def update_progress_bar(self,value):
        self.main_window.importProgressBar_center.setValue(value)

    def detectResults(self,frames):
        self.main_window.status_label_center.setText("[ SCANNING HUMANS ]")
        self.main_window.importProgressBar_center.setValue(0)
        self.human_detection_thread = HumanDetectionThread(
                                                            video_frames = frames,
                                                            isFront=False,
                                                            humanDetectionModel=self.human_detect_model,
                                                            humanDetectConf=self.human_detect_conf,
                                                        )
        self.human_detection_thread.human_track_results.connect(self.update_detection_results)
        self.human_detection_thread.human_detection_progress_update.connect(self.update_progress_bar)
        self.human_detection_thread.start()

    def update_detection_results(self, results_list):
        self.humanDetectionResults = results_list
        self.detect_keypoints(results_list)
    
    def detect_keypoints(self, humanDetectedResults):
        self.main_window.status_label_center.setText("[ DETECTING KEYPOINTS ]")
        self.pose_detection_thread = PoseDetectionThread(original_frames=self.returned_frames,
                                                         human_detect_results=humanDetectedResults,
                                                         humanDetectionModel=self.human_detect_model,
                                                         humanDetectConf=self.human_detect_conf,
                                                         humanPoseModel=self.human_pose_model,
                                                         humanPoseConf=self.human_pose_conf
                                                         )
        self.pose_detection_thread.pose_detection_results.connect(self.update_pose_detection_results)
        self.pose_detection_thread.pose_detection_progress_update.connect(self.update_progress_bar)  
        self.pose_detection_thread.start()
    
    def update_pose_detection_results(self, pose_results):
        self.humanPoseDetectionResults = pose_results
        #print(len(pose_results))
        #gENERATE wHITE FRAMES
        self.generate_white_frames()

    def generate_white_frames(self):
        self.video_processor.stop()
        self.main_window.status_label_center.setText("[ CREATING FRAMES ]")
        self.white_frame_generator = WhiteFrameGenerator(number_of_frames=self.number_of_frames,
                                                         height=1080,
                                                         width=1920)
        self.white_frame_generator.start()
        self.white_frame_generator.progress_update.connect(self.update_progress_bar)
        self.white_frame_generator.return_white_frames.connect(self.update_white_frame_list)
    
    def update_white_frame_list(self, frames):
        self.main_window.center_white_frames_preview = frames
        self.drawBoundingBoxes()
    

    def drawBoundingBoxes(self):
        self.main_window.status_label_center.setText("[ DRAWING BOXES ]")
        self.draw_bbox = DrawingBoundingBoxesThread(results=self.humanDetectionResults,
                                                    white_frames=self.main_window.center_white_frames_preview)
        self.draw_bbox.frame_drawn_list.connect(self.update_white_frame_list_then_draw_keypoints)
        self.draw_bbox.progress_updated.connect(self.update_progress_bar)
        self.draw_bbox.start()
    
    def update_white_frame_list_then_draw_keypoints(self, frames):
        self.main_window.center_white_frames_preview = frames
        self.main_window.status_label_center.setText("[ DRAWING KEYPOINTS ]")
        self.draw_keypoints = DrawingKeyPointsThread(video_frames=self.returned_frames,
                                                    white_frames=frames,
                                                      keypoints_list=self.humanPoseDetectionResults,
                                                      human_detections=self.humanDetectionResults)
        self.draw_keypoints.video_frame_drawn.connect(self.update_returned_frames_from_browsed_video)
        self.draw_keypoints.frame_drawn_list.connect(self.update_white_frame_list_then_identify_action)
        self.draw_keypoints.progress_updated.connect(self.update_progress_bar)
        self.draw_keypoints.start()

    def update_white_frame_list_then_identify_action(self, frames):
        self.main_window.front_white_frames_preview = frames
        self.identify_action()
    
    def update_white_frame_last(self, frames):
        self.main_window.status_label_center.setText("[ VIDEO IS READY! ]")
        self.main_window.center_white_frames_preview = frames
        self.main_window.play_pause_button_video_center.setEnabled(True)

        #Stop all Threads Running in order to save memory
        self.human_detection_thread.stop()
        self.pose_detection_thread.stop()
        self.white_frame_generator.stop()
        self.draw_bbox.stop()
        self.draw_keypoints.stop()
        self.action_detection_thread.stop()
    
    def closeEvent(self, event):
        self.video_processor.stop()
        self.white_frame_generator.stop()
        self.human_detection_thread.stop()
        event.accept()
    

    #=== FOR UPDATING FRAMES:

    def update_frame(self, frame):
        if frame is not None and self.main_window.is_center_video_playing:
            # Convert the frame from BGR to RGB
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.video_preview_label_center.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.main_window.video_preview_label_center.setPixmap(scaled_pixmap)

    def update_white_frame(self, white_frame):
        if white_frame is not None and self.main_window.is_center_video_playing:
            
            height, width, channel = white_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(white_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.video_preview_label_center.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.main_window.video_preview_label_center.setPixmap(scaled_pixmap)
    
    def show_next_frame(self):
        if self.video_processor and self.main_window.is_center_video_playing:
            self.video_processor.frame_processed.connect(self.main_window.update_frame)

    def update_frame_processing(self):
        fps = self.main_window.fps_loading_rate_slider.value()
        self.main_window.fps_flider_value = fps

    def toggle_play_pause(self):
        if self.main_window.is_center_video_playing:
            self.main_window.is_center_video_playing = False
            self.main_window.play_pause_button_video_center.setText("PLAY")
        else:
            self.main_window.is_center_video_playing = True
            self.main_window.play_pause_button_video_center.setText("PAUSE")
    
    def update_returned_frames_from_browsed_video(self, frames): #THIS IS THE LAST FUNCTION TO BE CALLED IN THE PROCESS
        self.main_window.returned_frames_from_browsed_center_video = frames 
        self.returned_frames = frames
        
    def identify_action(self):
        self.main_window.status_label_center.setText("[ IDENTIFYING ACTIONS...]")
        self.action_detection_thread = ActionDetectionThread(video_keypoints=self.humanPoseDetectionResults,
                                                            black_frames=self.main_window.center_white_frames_preview,
                                                            video_frames=self.returned_frames,
                                                            detections=self.humanDetectionResults)
        
        self.action_detection_thread.processed_frames_list.connect(self.update_returned_frames_from_browsed_video)
        self.action_detection_thread.processed_black_frames_list.connect(self.update_white_frame_last) 
        self.action_detection_thread.progress_update.connect(self.update_progress_bar)  
        self.action_detection_thread.start()