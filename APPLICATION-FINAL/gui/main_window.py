import sys
import cv2
import os
import numpy as np
import psutil
import GPUtil
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget, QButtonGroup
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate

from ultralytics import YOLO

from utils.drawing_utils import DrawingUtils
from gui import Ui_MainWindow
from trackers import PoseDetection
from utils import VideoProcessor, Tools, VideoUtils
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
        
        self.human_detect_results = None
        self.video_processor = None
        self.center_white_frames_preview = None
        self.front_white_frames_preview = None

        self.human_detection_results = []
        
        self.is_center_video_playing = False
        self.is_front_video_playing = False

        self.frame_processing_value = None
        
        self.import_video_button_center.clicked.connect(self.CenterVideo.browse_video)
        self.play_pause_button_video_center.clicked.connect(self.CenterVideo.toggle_play_pause)

        self.import_video_button_front.clicked.connect(self.FrontVideo.browse_video)
        self.play_pause_button_video_front.clicked.connect(self.FrontVideo.toggle_play_pause)

        #=====GET FOOTAGE ANALYTICS
        self.are_videos_ready = False
        self.whole_classroom_height = 1080
        self.front_starting_y = 750
        self.center_starting_y = 100

        self.analyze_video_button.clicked.connect(self.switch_to_analytics_tab)


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
        
        if self.is_center_video_playing:
            self.center_video_counter += self.timer.interval()
            if (self.center_video_counter >= self.center_video_interval):
                if self.center_video_frame_counter >= len(self.returned_frames_from_browsed_center_video):
                    self.center_video_frame_counter = 0 #Resets the playing of video
                
                if self.keypointsOnlyChkBox_Center.isChecked():
                    self.CenterVideo.update_white_frame(self.center_white_frames_preview[self.center_video_frame_counter])
                else:
                    self.CenterVideo.update_frame(self.returned_frames_from_browsed_center_video[self.center_video_frame_counter])
                
                self.center_video_counter = 0
                self.center_video_frame_counter += 1
        
        if self.is_front_video_playing:
            self.front_video_counter += self.timer.interval()
            if (self.front_video_counter >= self.front_video_interval):
                if self.front_video_frame_counter >= len(self.returned_frames_from_browsed_front_video):
                    self.front_video_frame_counter = 0 #Resets the playing of video
                
                if self.keypointsOnlyChkBox_front.isChecked():
                    self.FrontVideo.update_white_frame(self.front_white_frames_preview[self.front_video_frame_counter])
                else:
                    self.FrontVideo.update_frame(self.returned_frames_from_browsed_front_video[self.front_video_frame_counter])
                
                self.front_video_counter = 0
                self.front_video_frame_counter += 1
        
        #For cropped analytics
        if self.are_videos_ready:
            self.cropped_front_video_counter += self.timer.interval()
            #To save some memory, stop the videos.
            self.is_center_video_playing = False
            self.is_front_video_playing = False
            
            if (self.cropped_front_video_counter >= self.center_video_interval):
                #if the counter for center video frame exceeds or is equal to the length of the frame list,
                #it resets the videod
                if self.cropped_front_video_frame_counter >= len(self.returned_frames_from_browsed_front_video):
                    self.cropped_front_video_frame_counter = 0 #it resets here haha
                    self.cropped_center_video_frame_counter = 0 #it resets here haha
                
                #Insert here the code for updating frame for each video
                #the 2 videos must have same 

                #For front view
                self.AnalyticsTab.update_frame_for_front_video_label(self.returned_frames_from_browsed_front_video[self.cropped_front_video_frame_counter],
                                                                    starting_y= self.front_starting_y,
                                                                    whole_classroom_height=self.whole_classroom_height
                                                                    )
                
                #For center view
                self.AnalyticsTab.update_frame_for_center_video_label(self.returned_frames_from_browsed_center_video[self.cropped_center_video_frame_counter],
                                                                    center_starting_y=self.center_starting_y,
                                                                    front_starting_y=self.front_starting_y,
                                                                    )

                self.cropped_front_video_counter = 0 #reset the interval haha
                self.cropped_center_video_counter = 0

                self.cropped_front_video_frame_counter += 1 #increment to proceed to next frame lol
                self.cropped_center_video_frame_counter += 1 #increment to proceed to next frame lol
                



        if self.clock_counter >= self.clock_interval:
            self.CreateDataset.update_usage()
            self.CreateDataset.update_time()
            self.clock_counter = 0

        if self.toggle_record_label_counter >= self.toggle_record_label_interval:
            self.toggleLabelVisibility()
            self.toggle_record_label_counter = 0
        
    
            
    def center(self):
        # Get the screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        # Get the window geometry
        window_geometry = self.frameGeometry()
        # Move the window to the center of the screen
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    

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
        if (self.returned_frames_from_browsed_center_video and self.returned_frames_from_browsed_front_video) is not None:
            self.toggle_analytics_tab()
            self.MainTab.setCurrentIndex(1)
            self.are_videos_ready = True
            self.play_pause_button_analytics.setEnabled(True)
            self.play_pause_button_analytics.setText("PAUSE")
        else:
            self.show_warning_message(status_title="Error!",
                                      message= "Please import a complete set of footages.")