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
                                QButtonGroup,
                                QComboBox)

from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QImage, QPixmap

from superqt import QRangeSlider

from ultralytics import YOLO

from utils.drawing_utils import DrawingUtils



from trackers import PoseDetection
from utils import Tools, VideoUtils, VideoPlayerThread, SeekingVideoPlayerThread
from gui_commands import (CenterVideo,
                          FrontVideo,
                          CreateDataset,
                          AnalyticsTab,)

#UI Design
from gui import Ui_MainWindow 
from gui.LogVis import LogsTab                ## Double check : kapag walang .LogVis import circuit with MainWindow  // but this method works fine   



from PySide6.QtWidgets import QDialog


from .recording_window_UI import Ui_RecordingWindow  # Import the UI class

class RecordingWindow(QDialog):  # ✅ Use QDialog instead of QMainWindow
    def __init__(self):
        super().__init__()
        self.ui = Ui_RecordingWindow()
        self.ui.setupUi(self)  # ✅ Apply UI


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        #==== Create INSTANCES =========
        
        self.CenterVideo = CenterVideo(main_window=self)
        self.FrontVideo = FrontVideo(main_window=self)
        self.CreateDataset = CreateDataset(main_window=self)
        self.fps = 20
        self.latest_heatmap_frame = None
        self.heatmap_frame = None  # ✅ Define it before it's used
        self.AnalyticsTab = None

        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        self.video_utils = VideoUtils()
        
        
        self.videoWidth = None
        self.videoHeight = None

        self.fps_flider_value = 30
        self.returned_frames_from_browsed_center_video = None       ####
        self.returned_frames_from_browsed_front_video = None        ####

    
        self.cropped_frame_for_front_video_analytics_preview = None   ####    
        self.cropped_frame_for_center_video_analytics_preview = None  ####
        
        self.analyticsTab_index = 1
        self.createDatasetTab_index = 2
        self.tab_index = 3

        self.createDataset_tab = self.MainTab.widget(self.createDatasetTab_index)  # Index of the tab you want to hide
        self.createDataset_tab_index = self.createDatasetTab_index  # Index of the tab you want to hide
        self.createDataset_tab_title = self.MainTab.tabText(self.createDataset_tab_index)

        self.tab_index = self.tab_index
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
        
        self.human_pose_detection = PoseDetection(humanDetectionModel="yolov8x.pt",
                                                  humanDetectConf=self.human_detect_conf,
                                                  humanPoseModel="yolov8x-pose.pt",
                                                  humanPoseConf=self.human_pose_conf)
        
        self.human_detect_results_front = None
        self.human_detect_results_center = None
        self.human_detect_results_list_front = None 
        self.human_detect_results_list_center = None
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

        self.video_player_thread_preview_2 = None
        self.video_player_thread_logs = None               ## double check

        self.center_video_path = None
        self.front_video_path = None
        
        self.import_video_button_center.clicked.connect(self.CenterVideo.browse_video)

        self.play_pause_button_video_preview.clicked.connect(self.toggle_play_pause_preview)

        self.import_video_button_front.clicked.connect(self.FrontVideo.browse_video)

        self.recording_window_button.clicked.connect(self.open_recording_window)

        #=====GET FOOTAGE ANALYTICS
        self.are_videos_ready = False
        self.whole_classroom_height = 1080
        self.center_video_height = 700
        self.front_starting_y = 0
        self.center_starting_y = 0

        self.analyze_video_button.clicked.connect(self.handle_analyze_button_click)

        self.play_pause_button_analytics.clicked.connect(self.toggle_play_pause_analytics)
        self.play_pause_button_analytics_2.clicked.connect(self.toggle_play_pause_logs)
        
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
        self.sliderContainer_2 = self.findChild(QWidget, "timeFrameContainer_2")  ##
        

        # THIS IS FOR TIME RANGE VIDEO SEEKER
        self.timeFrameRangeSlider = QRangeSlider(Qt.Horizontal)
        self.timeFrameRangeSlider.setMinimum(0)  # Example duration
        self.timeFrameRangeSlider.setMaximum(100)  # Example duration
        self.timeFrameRangeSlider.setValue((20, 80))  # Default selected range
        self.timeFrameRangeSlider.setFixedHeight(40)

        self.timeFrameRangeSlider_2 = QRangeSlider(Qt.Horizontal)
        self.timeFrameRangeSlider_2 .setMinimum(0)  # Example duration
        self.timeFrameRangeSlider_2 .setMaximum(100)  # Example duration                #double check
        self.timeFrameRangeSlider_2 .setValue((20, 40))  # Default selected range
        self.timeFrameRangeSlider_2 .setFixedHeight(40)

        

        # Replace placeholder with actual QRangeSlider
        layout = QVBoxLayout(self.sliderContainer) ##
        layout.addWidget(self.timeFrameRangeSlider)

        layout = QVBoxLayout(self.sliderContainer_2) ##
        layout.addWidget(self.timeFrameRangeSlider_2)

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
        self.interval_slider.valueChanged.connect(self.updateIntervalLabel) ##!
        self.sequence_slider.valueChanged.connect(self.updateSequenceLabel) ##!
        
        # Set Column Names
        self.action_table.setHorizontalHeaderLabels(["Actions", "# of Recordings"])
        
        # Center the window on the screen
        self.center()
        
        #Set the date:
        current_date = QDate.currentDate().toString()
        
        self.day_label.setText(f"{current_date}")
        
        # self.timer = QTimer(self) #TIMER FOR ALL!!!
        # self.timer.timeout.connect(self.update_all_with_timers) #=======Update all and just have conditional statements
        # self.timer.start(10) 
        
        
    # def update_all_with_timers(self):
        
    #     self.clock_counter += self.timer.interval()
    #     self.toggle_record_label_counter += self.timer.interval()
                

    #     if self.clock_counter >= self.clock_interval:
    #         self.CreateDataset.update_usage()
    #         self.CreateDataset.update_time()
    #         self.clock_counter = 0

    #     if self.toggle_record_label_counter >= self.toggle_record_label_interval:
    #         self.toggleLabelVisibility()
    #         self.toggle_record_label_counter = 0

        
    
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
        # self.blink_timer.stop()
        self.status_label.setText("NOT RECORDING")
        self.status_label.setVisible(True)
        
    # def toggleLabelVisibility(self):
    #     self.status_label.setVisible(not self.status_label.isVisible())
        
    def updateIntervalLabel(self, value):
        self.interval_label.setText(str(value))   ##!
        
    def updateSequenceLabel(self, value):
        self.sequence_label.setText(str(value))   ##!
        
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
    def toggle_logs_tab(self): 

        if self.tab_index == -1:  
            self.MainTab.addTab(self.tab, "Logs")
            self.tab_index = self.MainTab.indexOf(self.tab)  
            self.MainTab.setCurrentIndex(self.tab_index)  
        else:  # If already present, remove it
            self.MainTab.removeTab(self.tab_index)
            self.tab_index = -1  
    def handle_analyze_button_click(self):
        self.switch_to_analytics_tab()  
        self.toggle_logs_tab()          


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

    def switch_to_analytics_tab(self):
        from gui import ActionVisualization
    
        
        
        if (self.is_center_video_ready and self.is_front_video_ready) is not None:
            self.toggle_analytics_tab()
            self.MainTab.setCurrentIndex(1)
            self.are_videos_ready = True
            self.play_pause_button_analytics.setEnabled(True)    
            self.play_pause_button_analytics.setText("PLAY")
            self.activate_analytics(activation=True)
            self.activate_Log()           ##!!!!
            self.play_pause_button_video_preview.setText("PLAY PREVIEW")
            self.import_video_button_front.setEnabled(False)
            self.import_video_button_center.setEnabled(False)

            self.play_pause_button_analytics_2.setEnabled(True)   
            self.play_pause_button_analytics_2.setText("PLAY")



            min_value, max_value = self.timeFrameRangeSlider.value()   ##!!
            min_value, max_value = self.timeFrameRangeSlider_2.value()  

            # Integrate chart into QGraphicsView instead of a separate window
            self.action_chart = ActionVisualization(main_window=self,
                                                    action_results_list_front=self.action_results_list_front,
                                                    action_results_list_center=self.action_results_list_center,
                                                    min_time=min_value,
                                                    max_time=max_value)
            self.AnalyticsTab = AnalyticsTab(
                                            main_window=self,
                                            action_results_front=self.action_results_list_front,
                                            action_results_center=self.action_results_list_center,
                                            human_detect_results_front = self.human_detect_results_front,
                                            human_detect_results_center = self.human_detect_results_center,
                                            center_video_path=self.center_video_path,
                                            front_video_path=self.front_video_path,
                                            ) 
            self.video_utils = VideoPlayerThread(
            main_window=self,
            center_video_path=self.center_video_path,
            front_video_path=self.front_video_path
                                    ) 
            
            # self.LogsVis = LogsTab(main_window=self, 
            #                     action_results_list_front=self.action_results_list_front or [], 
            #                     action_results_list_center=self.action_results_list_center or [],
            #                     min_time = min_value,
            #                     max_time = max_value 
            #                   )
            # self.LogsVis.row_selected.connect(self.update_video_position)


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

        if self.video_player_thread_preview is not None:
            if self.video_player_thread_preview.isRunning():
                self.video_player_thread_preview.running = False
                self.video_player_thread_preview.stop()
                
            else:
                self.video_player_thread_preview = None
                                                                                                                        # FInal final?
        #Just to stop the thread in viewing in order to save some CPU USAGE
        #self.video_player_thread_preview.stop()
        if self.video_player_thread_analytics is None:
            self.video_player_thread_analytics = SeekingVideoPlayerThread(center_video_path=center_video_directory,
                                                                           front_video_path=front_video_directory,
                                                                           main_window=self,
                                                                           selected_action = None, 
                                                                           filtered_bboxes_front = None, 
                                                                           filtered_bboxes_center= None,
                                                                            
                                                                            )
        # if self.video_player_thread_analytics is not None or self.video_player_thread_analytics.isRunning():
            self.play_pause_button_analytics.setText("PLAY")
            self.video_player_thread_analytics.frames_signal.connect(self.update_frame_for_analytics)
            self.video_player_thread_analytics.start()

            min_value, max_value = self.timeFrameRangeSlider.value()
            self.video_player_thread_analytics.current_frame_index = min_value
            self.video_player_thread_analytics.target_frame_index = min_value

        else:
            self.play_pause_button_analytics.setText("PAUSE")
            self.video_player_thread_analytics.pause(not self.video_player_thread_analytics.paused) 

    def extract_bounding_boxes(self, human_detect_results):
        """Extracts bounding boxes from human detection results."""
        
        if not human_detect_results or not isinstance(human_detect_results, list):

            return {}

        bbox_results = {}  # Store bounding boxes by person ID

        for idx, person_data in enumerate(human_detect_results):

            if isinstance(person_data, dict) and "track_id" in person_data and "bbox" in person_data:
                track_id = person_data["track_id"]  # Unique identifier
                bbox_results[track_id] = {"bbox": person_data["bbox"]}
            


        return bbox_results
    
    def update_frame_for_analytics(self, frame_list):
        if frame_list is None or len(frame_list) < 3:
            print("[ERROR] Frame list is None or incomplete!")
            return  # Avoid further errors
        if frame_list is not None:
            center_frame = frame_list[0]
            front_frame = frame_list[1]
            heatmap_frame = frame_list[2]

            self.AnalyticsTab.update_frame_for_center_video_label(
                frame=center_frame,
                center_starting_y=self.center_starting_y,                              ### 2nd to the last final
                front_starting_y=self.center_video_height
            )

            self.AnalyticsTab.update_frame_for_front_video_label(
                frame=front_frame,
                starting_y=self.front_starting_y,
                whole_classroom_height=self.whole_classroom_height
            )
            self.AnalyticsTab.update_heatmap(
            frame=heatmap_frame,
        )


            if self.human_detect_results_front is None or self.human_detect_results_center is None:
                print("[ERROR] Detection data is missing! Heatmap cannot be updated.")
                return  # Stop execution if detection data is missing


    def update_frame_for_logs(self, frame_list):
         if frame_list is None or len(frame_list) < 3:
            print("[ERROR] Frame list is None or incomplete!")
            return  # Avoid further errors

         if frame_list is not None:                                                       ### double check function
            center_frame = frame_list[0]
            front_frame = frame_list[1]

            # self.LogsVis.update_frame_for_center_video_label(
            #     frame=center_frame,
            #     center_starting_y=self.center_starting_y,                              ### double check
            #     front_starting_y=self.center_video_height
            # )

            # self.LogsVis.update_frame_for_front_video_label(
            #     frame=front_frame,
            #     starting_y=self.front_starting_y,
            #     whole_classroom_height=self.whole_classroom_height
            # ) 

    def toggle_play_pause_logs(self):
        center_video_directory = self.videoDirectory_center.text()
        front_video_directory = self.videoDirectory_front.text()

        # Stop preview thread to save CPU usage
        if self.video_player_thread_preview_2 is not None:
            if self.video_player_thread_preview_2.isRunning():
                self.video_player_thread_preview_2.running = False
                self.video_player_thread_preview_2.stop()
            self.video_player_thread_preview_2 = None  

        # If the log thread is None, initialize it and start playback
        if self.video_player_thread_logs is None:
            self.video_player_thread_logs = VideoPlayerThread(
                center_video_path=center_video_directory,
                front_video_path=front_video_directory,
                main_window=self
            )

            self.video_player_thread_logs.frames_signal.connect(self.update_frame_for_logs)
            self.video_player_thread_logs.start()

            min_value, max_value = self.timeFrameRangeSlider_2.value()
            self.video_player_thread_logs.current_frame_index = min_value
            self.video_player_thread_logs.target_frame_index = min_value

            self.play_pause_button_analytics_2.setText("PAUSE")  # Change button text to indicate it's playing

        else:
            # Toggle between pause and resume
            if self.video_player_thread_logs.paused:
                self.video_player_thread_logs.pause(False)  # Resume playback
                self.play_pause_button_analytics_2.setText("PAUSE")
            else:
                self.video_player_thread_logs.pause(True)  # Pause playback
                self.play_pause_button_analytics_2.setText("PLAY")



    def toggle_play_pause_analytics_2(self):
        """ Toggles play/pause for the logs in the Logs tab. """    ## i think this is useless?
        # self.LogsVis.toggle_play_pause_logs()
    

    '''
    THIS AREA IS FOR VIDEO PLAYER THREAD
    '''
    def update_frame_for_preview(self, frame_list):
        '''
        This function is for updating the picture on the frames.
        More like key component to show frames and to look like a video

        '''
        if frame_list is not None:
            
            center_frame = frame_list[0]
            front_frame = frame_list[1]

            '''
            THIS AREA IS FOR CENTER FRAME
            '''
            # Convert the frame from BGR to RGB
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            

            '''
            THIS AREA IS FOR CENTER FRAME
            '''
            self.CenterVideo.update_frame(center_frame=center_frame)
            '''
            THIS AREA IS FOR FRONT FRAME
            '''
            self.FrontVideo.update_frame(front_frame=front_frame)

            

    '''
    THIS AREA IS FOR VIDEO PLAYER THREAD
    '''

    def activate_Log(self):
            front_video_directory = self.videoDirectory_front.text()
            center_video_directory = self.videoDirectory_center.text()
            
            front_cap = cv2.VideoCapture(front_video_directory)
            center_cap = cv2.VideoCapture(center_video_directory)

            front_video_frame_count = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            center_video_frame_count = int(center_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if front_video_frame_count > center_video_frame_count:
                self.timeFrameRangeSlider_2.setMaximum(int(len(self.human_pose_results_center)-1)) ### double check function
            else:
                self.timeFrameRangeSlider_2.setMaximum(int(len(self.human_pose_results_front)-1))


    def activate_analytics(self, activation):
            front_video_directory = self.videoDirectory_front.text()
            center_video_directory = self.videoDirectory_center.text()
                                                                                            #### Intimidate
            front_cap = cv2.VideoCapture(front_video_directory)
            center_cap = cv2.VideoCapture(center_video_directory)

            front_video_frame_count = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            center_video_frame_count = int(center_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if front_video_frame_count > center_video_frame_count:
                self.timeFrameRangeSlider.setMaximum(int(len(self.human_pose_results_center)-1)) ##!!
            else:
                self.timeFrameRangeSlider.setMaximum(int(len(self.human_pose_results_front)-1))

            self.play_pause_button_video_preview.setEnabled(activation)
            self.analyze_video_button.setEnabled(activation)


    
    def open_recording_window(self):
        self.recording_window_button.setDisabled(True)
        self.recording_window = RecordingWindow()
        self.recording_window.finished.connect(self.reenable_recording_button)
        self.recording_window.show()
    
    def reenable_recording_button(self):
        self.recording_window_button.setDisabled(False)

    def update_video_position(self, timestamp):
        print(f"Double-clicked row, emitting timestamp {timestamp} seconds")  # Already in seconds ✅

        timestamp_seconds = float(timestamp)  # Ensure it's a float

        # Get current slider range
        slider_value = self.timeFrameRangeSlider_2.value()
        print(f"Slider current value: {slider_value}, Type: {type(slider_value)}")

        if isinstance(slider_value, tuple):  # If it's a range slider
            min_time, max_time = slider_value
            print(f"Slider current range: {min_time} - {max_time}")

            # Convert timestamp (seconds) to a frame index
            new_slider_value = int(timestamp_seconds * self.fps)

            # Ensure it stays within valid range
            new_slider_value = max(min_time, min(new_slider_value, max_time))

            # Update only the minimum time of the range slider
            self.timeFrameRangeSlider_2.setValue((new_slider_value, max_time))
        else:  # If it's a single-value slider
            min_time = self.timeFrameRangeSlider_2.minimum()
            max_time = self.timeFrameRangeSlider_2.maximum()

            # Convert timestamp to a valid slider position
            new_slider_value = int(timestamp_seconds * self.fps)

            # Clamp value within valid slider range
            new_slider_value = max(min_time, min(new_slider_value, max_time))

            self.timeFrameRangeSlider_2.setValue(new_slider_value)  # Set integer value

        print(f"✅ Slider updated to: {self.timeFrameRangeSlider_2.value()} (converted from {timestamp_seconds} sec)")






