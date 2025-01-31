import cv2

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap



from utils import (VideoProcessorThread, VideoPlayerThread)
from trackers import (PoseDetection,
                    ActionDetectionThread)

class CenterVideo:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.human_detect_model = "yolov8n.pt"
        self.human_detect_conf = 0.5
        self.human_pose_model = "yolov8n-pose.pt"
        self.human_pose_conf = 0.5
        self.iou_value = 0.3

        self.returned_frames = []
        self.humanDetectionResults = []
        self.action_results_list = []

        self.humanPoseDetectionResults = None
        self.isImportDone = False
        self.videoHeight = None
        self.videoWidth = None
        self.number_of_frames = 0

        self.video_player_thread = None

        self.directory = None
    
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
        self.directory, _ = QFileDialog.getOpenFileName(self.main_window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if self.directory:
            self.main_window.videoDirectory_center.setText(f"{self.directory}")
            self.start_video_processing(self.directory)
    
    def start_video_processing(self, video_path):
        self.main_window.status_label_center.setText("[ PROCESSING VIDEO... PLEASE WAIT ]")

        self.video_processor = VideoProcessorThread(video_path, resize_frames=False,
                                                    isFront=False,
                                                    human_detection_model=self.human_detect_model,
                                                    human_detection_confidence=self.human_detect_conf,
                                                    human_pose_model=self.human_pose_model,
                                                    human_pose_confidence=self.human_pose_conf,
                                                    main_window=self.main_window)
        
        self.video_processor.human_detect_results.connect(self.update_detection_results)
        self.video_processor.human_pose_results.connect(self.update_pose_detection_results)
        self.video_processor.progress_update.connect(self.update_progress_bar)

        #Start the operation
        self.video_processor.start()
        
    def update_detection_results(self, results_list):
        self.humanDetectionResults = results_list
        self.main_window.human_detect_results_center = results_list

        print("HUMAN DETECT RESULTS:", len(results_list))
    
    def update_pose_detection_results(self, pose_results):
        self.humanPoseDetectionResults = pose_results
        self.main_window.human_pose_results_center = pose_results
        self.identify_actions()

        print("POSE RESULTS:", len(pose_results))
    
    def identify_actions(self):
        self.main_window.status_label_center.setText("[ IDENTIFYING ACTIONS...]")

        self.action_detection_thread = ActionDetectionThread(video_keypoints=self.humanPoseDetectionResults,
                                                            black_frames=self.main_window.center_white_frames_preview,
                                                            video_frames=self.returned_frames,
                                                            detections=self.humanDetectionResults)
        
        self.action_detection_thread.detected_actions_list.connect(self.update_action_results)
        self.action_detection_thread.progress_update.connect(self.update_progress_bar)  
        self.action_detection_thread.start()

    def update_action_results(self, actions):
        self.action_results_list = actions
        self.main_window.action_results_list_center = actions
        self.main_window.status_label_center.setText("[ ACTIONS IDENTIFICATION, DONE! ]")
        self.main_window.play_pause_button_video_center.setEnabled(True)
    
    def update_progress_bar(self,value):
        self.main_window.importProgressBar_center.setValue(value)

    def closeEvent(self, event):
        self.video_processor.stop()
        self.white_frame_generator.stop()
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
        """Start, pause, or resume video playback."""
        if self.video_player_thread is None or not self.video_player_thread.isRunning():
            self.video_player_thread = VideoPlayerThread(video_path=self.directory,
                                                        main_window=self.main_window,
                                                        is_Front=False,
                                                        target_frame_index=0)
            self.video_player_thread.frame_signal.connect(self.update_frame)
            self.video_player_thread.start()
            self.main_window.play_pause_button_video_center.setText("PLAY")
        else:
            self.video_player_thread.pause(not self.video_player_thread.paused) 
            self.main_window.play_pause_button_video_center.setText("PAUSE")

    def pause(self, status):
        """Pause or resume the video playback."""
        self.paused = status


    def start_video(self):
        if self.video_player_thread is None or not self.video_player_thread.isRunning():
            self.video_player_thread = VideoPlayerThread(self.directory)
            self.video_player_thread.frame_signal.connect(self.update_frame)
            self.video_player_thread.finished.connect(self.cleanup_thread)
            self.video_player_thread.start()

    def update_frame(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.main_window.video_preview_label_center.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.main_window.video_preview_label_center.setPixmap(scaled_pixmap)

    def stop_video(self):
        if self.video_player_thread and self.video_player_thread.isRunning():
            self.video_player_thread.stop()

    def cleanup_thread(self):
        """Called when the thread finishes to release memory."""
        self.video_player_thread.deleteLater()
        self.video_player_thread = None