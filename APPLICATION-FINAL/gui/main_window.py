import sys
import cv2
import os
import numpy as np
import psutil
import GPUtil
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget, QButtonGroup
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QScreen, QImage, QPixmap

from PySide6.QtMultimediaWidgets import QVideoWidget

from pygrabber.dshow_graph import FilterGraph

from ultralytics import YOLO

from utils.camera import CameraFeed
from utils.drawing_utils import DrawingUtils
from gui import Ui_MainWindow
from trackers import PoseDetection
from utils import VideoProcessor, Tools, VideoUtils
from gui_commands import (CenterVideo,
                          FrontVideo,
                          CreateDataset)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        #==== Create INSTANCES =========
        
        self.CenterVideo = CenterVideo(main_window=self)
        self.FrontVideo = FrontVideo(main_window=self)
        self.CreateDataset = CreateDataset(main_window=self)
        
        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        self.video_utils = VideoUtils()
        
        self.videoWidth = None
        self.videoHeight = None

        self.fps_flider_value = 30
        self.returned_frames_from_browsed_center_video = None
        self.returned_frames_from_browsed_front_video = None
        
        self.analyticsTab_index = 1
        self.createDatasetTab_index = 2
        self.createDataset_tab = self.MainTab.widget(self.createDatasetTab_index)  # Index of the tab you want to hide
        self.createDataset_tab_index = self.createDatasetTab_index  # Index of the tab you want to hide
        self.createDataset_tab_title = self.MainTab.tabText(self.createDataset_tab_index)

        # Initially hide the tab
        self.MainTab.removeTab(self.createDataset_tab_index)
        
        self.analytics_tab_title = self.MainTab.tabText(self.analyticsTab_index) 
        self.analytics_tab_index = self.analyticsTab_index
        self.analytics_tab_title = self.MainTab.tabText(self.analytics_tab_index)

        # Initially hide the tab
        self.MainTab.removeTab(self.analytics_tab_index)
        
        

        # Connect the QAction's triggered signal to the toggle_tab method
        self.actionviewCreateDataset.triggered.connect(self.toggle_tab)

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
        self.center_white_frames_preview = []
        self.front_white_frames_preview = []

        self.human_detection_results = []
        
        self.is_center_video_playing = False
        self.is_front_video_playing = False

        self.frame_processing_value = None
        
        self.import_video_button_cernter.clicked.connect(self.CenterVideo.browse_video)
        self.play_pause_button_video_center.clicked.connect(self.CenterVideo.toggle_play_pause)

        self.import_video_button_front.clicked.connect(self.FrontVideo.browse_video)
        self.play_pause_button_video_front.clicked.connect(self.FrontVideo.toggle_play_pause)

        #Adjust this according to video but meh. This is just the default. Just check the specs of the cam. Default naman sya all the times
        self.center_video_interval = 1000//30 #30 fps
        self.front_video_interval = 1000//30 #30 fps
        self.clock_interval = 1000  # 1 second interval for clock
        self.toggle_record_label_interval = 750

        self.center_video_counter = 0
        self.front_video_counter = 0
        
        self.clock_counter = 0
        self.toggle_record_label_counter = 0
        self.toggle_import_indicator = 0
        self.center_video_frame_counter = 0        
        self.front_video_frame_counter = 0    

        
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
        
    def toggle_tab(self):
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