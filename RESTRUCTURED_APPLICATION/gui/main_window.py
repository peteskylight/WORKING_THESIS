import sys
import cv2
import os
import numpy as np
import psutil
import GPUtil
from PySide6.QtWidgets import (QApplication,
                                QMainWindow,
                                QVBoxLayout,
                                QFileDialog,
                                QMessageBox,
                                QTableWidgetItem,
                                QWidget,
                                QButtonGroup)

from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QImage, QPixmap

from superqt import QRangeSlider

from ultralytics import YOLO

from utils.drawing_utils import DrawingUtils
from gui import Ui_MainWindow
from trackers import PoseDetection
from utils import Tools, VideoUtils, VideoPlayerThread, SeekingVideoPlayerThread
from gui_commands import (CenterVideo,
                          FrontVideo,
                          CreateDataset,
                          AnalyticsTab)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        #==== Create INSTANCES =========
        
        self.CenterVideo = CenterVideo(main_window=self)
        self.FrontVideo = FrontVideo(main_window=self)
        self.CreateDataset = CreateDataset(main_window=self)
        self.AnalyticsTab = AnalyticsTab(main_window=self)
        
        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        self.video_utils = VideoUtils()
        
        self.videoWidth = None
        self.videoHeight = None

        self.fps_flider_value = 30
        self.returned_frames_from_browsed_center_video = None
        self.returned_frames_from_browsed_front_video = None

        

        self.cropped_frame_for_front_video_analytics_preview = None
        self.cropped_frame_for_center_video_analytics_preview = None
        
        self.analyticsTab_index = 1
        self.createDatasetTab_index = 2

        self.createDataset_tab = self.MainTab.widget(self.createDatasetTab_index)  # Index of the tab you want to hide
        self.createDataset_tab_index = self.createDatasetTab_index  # Index of the tab you want to hide
        self.createDataset_tab_title = self.MainTab.tabText(self.createDataset_tab_index)

        # Initially hide the tab
        self.MainTab.removeTab(self.createDataset_tab_index)
        
        self.analytics_tab = self.MainTab.widget(self.analyticsTab_index) 
        self.analytics_tab_index = self.analyticsTab_index
        self.analytics_tab_title = self.MainTab.tabText(self.analytics_tab_index)

        # Initially hide the tab
        self.MainTab.removeTab(self.analytics_tab_index)
        
        # Connect the QAction's triggered signal to the toggle_createDataset_tab method
        self.actionviewCreateDataset.triggered.connect(self.toggle_createDataset_tab)

        #============ FOR IMPORTING VIDEO TAB ===========
        
        #Create INSTANCES\
        self.human_detect_model = YOLO("yolov8n.pt")
        self.human_pose_model = YOLO("yolov8n-pose.pt")
        self.human_detect_conf = 0.5
        self.human_pose_conf = 0.5
        
        self.human_pose_detection = PoseDetection(humanDetectionModel="yolov8n.pt",
                                                  humanDetectConf=self.human_detect_conf,
                                                  humanPoseModel="yolov8n-pose.pt",
                                                  humanPoseConf=self.human_pose_conf)
        
        self.human_detect_results_front = None
        self.human_detect_results_center = None
        self.human_pose_results_front = None
        self.human_pose_results_center = None
        
        self.action_results_list_front = None
        self.action_results_list_center = None

        self.center_white_frames_preview = None
        self.front_white_frames_preview = None

        self.is_center_video_ready = False
        self.is_front_video_ready = False

        self.frame_processing_value = None

        self.video_player_thread_preview = None
        self.video_player_thread_analytics = None
        
        self.import_video_button_center.clicked.connect(self.CenterVideo.browse_video)

        self.play_pause_button_video_preview.clicked.connect(self.toggle_play_pause_preview)

        self.import_video_button_front.clicked.connect(self.FrontVideo.browse_video)

        #=====GET FOOTAGE ANALYTICS
        self.are_videos_ready = False
        self.whole_classroom_height = 1080
        self.front_starting_y = 750
        self.center_starting_y = 100

        self.analyze_video_button.clicked.connect(self.switch_to_analytics_tab)

        self.play_pause_button_analytics.clicked.connect(self.toggle_play_pause_analytics)

        # self.video_seek_slider.sliderPressed.connect(self.on_slider_clicked)
        # self.video_seek_slider.sliderReleased.connect(self.on_slider_moved)

        #Adjust this according to video but meh. This is just the default. Just check the specs of the cam. Default naman sya all the times
        self.center_video_interval = 1000//30 #30 fps
        self.front_video_interval = 1000//30 #30 fps

        self.clock_interval = 1000  # 1 second interval for clock
        self.toggle_record_label_interval = 750

        self.center_video_counter = 0
        self.front_video_counter = 0
        self.cropped_center_video_counter = 0
        self.cropped_front_video_counter = 0

        
        self.clock_counter = 0
        self.toggle_record_label_counter = 0
        self.toggle_import_indicator = 0
        self.center_video_frame_counter = 0        
        self.front_video_frame_counter = 0

        self.cropped_center_video_frame_counter = 0
        self.cropped_front_video_frame_counter = 0

        self.time_range_start_time = 0
        self.time_range_end_time = 0

        # Locate the QWidget placeholder
        self.sliderContainer = self.findChild(QWidget, "timeFrameContainer")
        

        # THIS IS FOR TIME RANGE VIDEO SEEKER
        self.timeFrameRangeSlider = QRangeSlider(Qt.Horizontal)
        self.timeFrameRangeSlider.setMinimum(0)  # Example duration
        self.timeFrameRangeSlider.setMaximum(100)  # Example duration
        self.timeFrameRangeSlider.setValue((20, 80))  # Default selected range
        self.timeFrameRangeSlider.setFixedHeight(40)

        # Replace placeholder with actual QRangeSlider
        layout = QVBoxLayout(self.sliderContainer)
        layout.addWidget(self.timeFrameRangeSlider)

        #self.timeFrameRangeSlider.valueChanged.connect(self.update_time_range)

        #============ FOR ANALYTICS TAB =================
        

        #============ FOR CREATE DATASET TAB ============
        #Disable Maximize Button
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)

        self.closeCamera.clicked.connect(self.CreateDataset.stop_camera)
        self.openCamera.clicked.connect(self.CreateDataset.start_camera)
        self.browseButton.clicked.connect(self.browse_button_functions)
        self.refresh_button.clicked.connect(lambda: self.CreateDataset.scan_directory(self.directoryLineEdit.text()))
        self.add_action_button.clicked.connect(self.CreateDataset.add_folder)
        self.delete_action_button.clicked.connect(self.CreateDataset.delete_folder)
        self.recording_button.clicked.connect(self.toggle_button)
        self.refresh_action_list.clicked.connect(self.CreateDataset.showActionsToTable)
        self.CreateDataset.populate_camera_combo_box()

        # Slider Value Change
        self.interval_slider.valueChanged.connect(self.updateIntervalLabel)
        self.sequence_slider.valueChanged.connect(self.updateSequenceLabel)
        
        # Set Column Names
        self.action_table.setHorizontalHeaderLabels(["Actions", "# of Recordings"])
        
        # Center the window on the screen
        self.center()
        
        #Set the date:
        current_date = QDate.currentDate().toString()
        
        self.day_label.setText(f"{current_date}")
        
        self.timer = QTimer(self) #TIMER FOR ALL!!!
        self.timer.timeout.connect(self.update_all_with_timers) #=======Update all and just have conditional statements
        self.timer.start(10) 
        
    def update_all_with_timers(self):
        
        self.clock_counter += self.timer.interval()
        self.toggle_record_label_counter += self.timer.interval()
                

        if self.clock_counter >= self.clock_interval:
            self.CreateDataset.update_usage()
            self.CreateDataset.update_time()
            self.clock_counter = 0

        if self.toggle_record_label_counter >= self.toggle_record_label_interval:
            self.toggleLabelVisibility()
            self.toggle_record_label_counter = 0
        
    
    def center(self):
        '''
        This centers the appearance of the window on the screen.
        '''
        # Get the screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        # Get the window geometry
        window_geometry = self.frameGeometry()
        # Move the window to the center of the screen
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    #THIS IS FOR CREATE DATASET
    '''
    THIS AREA NEEDS MAINTENANCE!!!
    '''
    def browse_button_functions(self):
        self.CreateDataset.open_file_explorer()
        self.CreateDataset.showActionsToTable()

    def startBlinking(self):
        self.status_label.setStyleSheet("""
                QLabel {
                    color: rgb(0, 255, 0);
                }
            """)
        self.status_label.setVisible(True) # Ensure the label is visible when starting
    
    def stopBlinking(self):
        self.status_label.setStyleSheet("""
                QLabel {
                    color: rgb(0, 255, 0);
                }
            """)
        self.blink_timer.stop()
        self.status_label.setText("NOT RECORDING")
        self.status_label.setVisible(True)
        
    def toggleLabelVisibility(self):
        self.status_label.setVisible(not self.status_label.isVisible())
        
    def updateIntervalLabel(self, value):
        self.interval_label.setText(str(value))
        
    def updateSequenceLabel(self, value):
        self.sequence_label.setText(str(value))
        
    def toggle_createDataset_tab(self):
        # Toggle the visibility of the tab
        if self.createDataset_tab_index == -1 or not self.MainTab.isTabVisible(self.createDataset_tab_index):
            # Add the tab back
            self.MainTab.addTab(self.createDataset_tab, self.createDataset_tab_title)
            self.createDataset_tab_index = self.MainTab.indexOf(self.createDataset_tab)
            self.MainTab.setCurrentIndex(self.createDataset_tab_index)
        else:
            # Remove the tab
            self.MainTab.removeTab(self.createDataset_tab_index)
            self.createDataset_tab_index = -1
    
    def toggle_analytics_tab(self):
        # Toggle the visibility of the tab
        if self.analytics_tab_index == -1 or not self.MainTab.isTabVisible(self.analytics_tab_index):
            # Add the tab back
            self.MainTab.addTab(self.analytics_tab, self.analytics_tab_title)
            self.analytics_tab_index = self.MainTab.indexOf(self.analytics_tab)
            self.MainTab.setCurrentIndex(self.analytics_tab_index)
        else:
            # Remove the tab
            self.MainTab.removeTab(self.analytics_tab_index)
            self.analytics_tab_index = -1

    def toggle_button(self):
        if self.recording_button.text() == "START\nRECORDING":
            self.recording_button.setText("STOP\nRECORDING")
            self.recording_button.setStyleSheet("""
                QPushButton {
                    background-color: rgb(170, 0, 0);
                    border-radius: 15px; /* Adjust the radius as needed */
                    color: black; /* Set the text color */
                    border: 1px solid black; /* Optional: Add a border */
                }
            """)
            self.status_label.setText("RECORDING")
            self.startBlinking()
            
        else:
            self.recording_button.setText("START\nRECORDING")
            self.recording_button.setStyleSheet("""
                QPushButton {
                    background-color: rgb(170, 255, 127);
                    border-radius: 15px; /* Adjust the radius as needed */
                    color: black; /* Set the text color */
                    border: 1px solid black; /* Optional: Add a border */
                }
            """)
            self.status_label.setText("NOT RECORDING")
    
    
    #template for warning message boxes:
    def show_warning_message(self, status_title, message):
        # Create and show a warning message box
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(status_title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    #Switch to analytics tab
    def switch_to_analytics_tab(self):
        if (self.is_center_video_ready and self.is_front_video_ready) is not None:
            self.toggle_analytics_tab()
            self.MainTab.setCurrentIndex(1)
            self.are_videos_ready = True
            self.play_pause_button_analytics.setEnabled(True)
            self.play_pause_button_analytics.setText("PLAY")

            self.play_pause_button_video_preview.setText("PLAY PREVIEW")
            self.import_video_button_front.setEnabled(False)
            self.import_video_button_center.setEnabled(False)
            #self.video_player_thread_preview.pause(True)
        else:
            self.show_warning_message(status_title="Error!",
                                      message= "Please import a complete set of footages.")

    '''
    THIS AREA IS FOR VIDEO PLAYER THREAD
    '''

    def toggle_play_pause_preview(self):
        """Start, pause, or resume video playback."""
        center_video_directory = self.videoDirectory_center.text()
        front_video_directory = self.videoDirectory_front.text()
        
        if self.video_player_thread_analytics is not None:
            if self.video_player_thread_analytics.isRunning():
                self.video_player_thread_analytics.running = False
                self.video_player_thread_analytics.stop()
            else:
                self.video_player_thread_analytics = None

        if self.video_player_thread_preview is None or not self.video_player_thread_preview.isRunning():
            self.keypointsOnlyChkBox_Center_analytics.setChecked(False)
            self.keypointsOnlyChkBox_front_analytics.setChecked(False)
            self.video_player_thread_preview = VideoPlayerThread(center_video_path=center_video_directory,
                                                         front_video_path= front_video_directory,
                                                         main_window=self)
            
            self.video_player_thread_preview.frames_signal.connect(self.update_frame_for_preview)
            self.video_player_thread_preview.start()
            self.play_pause_button_video_preview.setText("PLAY PREVIEW")
            self.import_video_button_front.setEnabled(False)
            self.import_video_button_center.setEnabled(False)
        else:
            self.play_pause_button_video_preview.setText("PAUSE PREVIEW")
            self.import_video_button_front.setEnabled(True)
            self.import_video_button_center.setEnabled(True)
            self.video_player_thread_preview.pause(not self.video_player_thread_preview.paused) 
            


    def toggle_play_pause_analytics(self):
        center_video_directory = self.videoDirectory_center.text()
        front_video_directory = self.videoDirectory_front.text()
        
        #Just to make sure that the frames that will be shown is not black frames

        self.keypointsOnlyChkBox_Center.setChecked(False)
        self.keypointsOnlyChkBox_front.setChecked(False)

        if self.video_player_thread_preview is not None:
            if self.video_player_thread_preview.isRunning():
                self.video_player_thread_preview.running = False
                self.video_player_thread_preview.stop()
                
            else:
                self.video_player_thread_preview = None

        #Just to stop the thread in viewing in order to save some CPU USAGE
        #self.video_player_thread_preview.stop()
        if self.video_player_thread_analytics is None or not self.video_player_thread_analytics.isRunning():
            self.keypointsOnlyChkBox_Center.setChecked(False)
            self.keypointsOnlyChkBox_front.setChecked(False)
            self.video_player_thread_analytics = SeekingVideoPlayerThread(center_video_path=center_video_directory,
                                                                           front_video_path=front_video_directory,
                                                                           main_window=self
                                                                            )
        # if self.video_player_thread_analytics is not None or self.video_player_thread_analytics.isRunning():
            self.play_pause_button_analytics.setText("PLAY")
            self.video_player_thread_analytics.frames_signal.connect(self.update_frame_for_analytics)
            self.video_player_thread_analytics.start()
        else:
            self.play_pause_button_analytics.setText("PAUSE")
            self.video_player_thread_analytics.pause(not self.video_player_thread_analytics.paused) 
    
    def update_frame_for_analytics(self, frame_list):
        if frame_list is not None:
            center_frame = frame_list[0]
            front_frame = frame_list[1]

            self.AnalyticsTab.update_frame_for_center_video_label(frame = center_frame,
                                                                  center_starting_y=self.center_starting_y,
                                                                  front_starting_y=self.front_starting_y)
            print("NAKAABOT HERE")
            self.AnalyticsTab.update_frame_for_front_video_label(frame=front_frame,
                                                                 starting_y=self.front_starting_y,
                                                                 whole_classroom_height=self.whole_classroom_height
                                                                 )

    '''
    THIS AREA IS FOR VIDEO PLAYER THREAD
    '''
    def update_frame_for_preview(self, frame_list):
        '''
        This function is for updating the picture on the frames.
        More like key component to show frames and to look like a video

        frame_list[0] - Center Video Frame
        frame_list[1] - Front Video Frame
        '''
        if frame_list is not None:
            
            center_frame = frame_list[0]
            front_frame = frame_list[1]

            '''
            THIS AREA IS FOR CENTER FRAME
            '''
            # Convert the frame from BGR to RGB
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if len(center_frame.shape) == 3:
                center_video_height, center_video_width, _ = center_frame.shape
            else:
                center_video_height, center_video_width = center_frame.shape

            bytes_per_line = 3 * center_video_width
            q_img_center = QImage(center_frame.data, center_video_width, center_video_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap_center = QPixmap.fromImage(q_img_center)
            scaled_pixmap_center = pixmap_center.scaled(self.video_preview_label_center.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.video_preview_label_center.setPixmap(scaled_pixmap_center)

            '''
            THIS AREA IS FOR CENTER FRAME
            '''

            '''
            THIS AREA IS FOR FRONT FRAME
            '''

            if len(front_frame.shape) == 3:
                front_video_height, front_video_width, _ = front_frame.shape
            else:
                front_video_height, front_video_width = front_frame.shape

            bytes_per_line = 3 * front_video_width
            q_img_front = QImage(front_frame.data, front_video_width, front_video_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap_front = QPixmap.fromImage(q_img_front)
            scaled_pixmap_front = pixmap_front.scaled(self.video_preview_label_front.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.video_preview_label_front.setPixmap(scaled_pixmap_front)

    '''
    THIS AREA IS FOR VIDEO PLAYER THREAD
    '''
    def activate_analytics(self, activation):
        self.timeFrameRangeSlider.setMaximum(int(len(self.human_pose_results_center)-1))
        self.play_pause_button_video_preview.setEnabled(activation)
        self.analyze_video_button.setEnabled(activation)
    