import sys
import os
import numpy as np
import psutil
import GPUtil
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QWidget
from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt
from PySide6.QtGui import QScreen

from pygrabber.dshow_graph import FilterGraph

from utils.camera import CameraFeed
from utils.drawing_utils import DrawingUtils
from gui import Ui_MainWindow
from trackers import PoseDetection

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)
        
        #Hide the create dataset tab
        #self.actionviewCreateDataset.triggered.connect(self.toggleCreateDatasetTab)
        
        self.createDatasetTab_index = 2
        self.hidden_tab = self.MainTab.widget(self.createDatasetTab_index)  # Index of the tab you want to hide
        self.hidden_tab_index = self.createDatasetTab_index  # Index of the tab you want to hide
        self.hidden_tab_title = self.MainTab.tabText(self.hidden_tab_index)
        # Initially hide the tab
        self.MainTab.removeTab(self.hidden_tab_index)
        # Connect the QAction's triggered signal to the toggle_tab method
        self.actionviewCreateDataset.triggered.connect(self.toggle_tab)

        
        #Disable Maximize Button
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)

        self.isRecording = False

        self.camera_feed_instance = CameraFeed(self.camera_feed, self.white_frame_feed, self)
        self.drawing_utils = DrawingUtils()
        
        self.closeCamera.clicked.connect(self.camera_feed_instance.stop_camera)
        
        self.openCamera.clicked.connect(self.start_camera)
        
        self.browseButton.clicked.connect(self.browse_button_functions)
        
        self.refresh_button.clicked.connect(lambda: self.scan_directory(self.directoryLineEdit.text()))
        
        self.add_action_button.clicked.connect(self.add_folder)
        
        self.delete_action_button.clicked.connect(self.delete_folder)
        
        self.recording_button.clicked.connect(self.toggle_button)
        
        self.refresh_action_list.clicked.connect(self.showActionsToTable)
        
        self.populate_camera_combo_box()

        # Slider Value Change
        self.interval_slider.valueChanged.connect(self.updateIntervalLabel)
        self.sequence_slider.valueChanged.connect(self.updateSequenceLabel)
        
        # Set Column Names
        self.action_table.setHorizontalHeaderLabels(["Actions", "# of Recordings"])
        
        # Center the window on the screen
        self.center()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_usage)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.toggleLabelVisibility)
        
    def start_camera(self):
        selected_index = self.cameraComboBox.currentIndex()
        self.camera_feed_instance.start_camera(selected_index)
    
    def populate_camera_combo_box(self):
        # List available cameras
        available_cameras = self.list_available_cameras()
        for camera in available_cameras:
            self.cameraComboBox.addItem(camera)

    def list_available_cameras(self):
        graph = FilterGraph()
        available_cameras = graph.get_input_devices()
        return available_cameras
    
    def update_usage(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        
        self.cpu_label.setText(f"{cpu_usage} %")
        self.ram_label.setText(f"{ram_usage} %")
        self.gpu_label.setText(f"{gpu_usage:.2f} %")
    
    def update_time(self):
        current_time = QTime.currentTime()
        self.timeLCD.display(current_time.toString("hh:mm:ss"))
        
    def open_file_explorer(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directoryLineEdit.setText(directory)
            self.scan_directory(directory)
        
    def scan_directory(self, directory):
        self.action_comboBox.clear()
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                self.action_comboBox.addItem(folder_name)

    def add_folder(self):
        directory = self.directoryLineEdit.text()
        folder_name = self.action_comboBox.currentText()  # Assuming you have a QLineEdit for folder name input

        if not os.path.isdir(directory):
            QMessageBox.critical(self, "Error", "The specified directory does not exist.")
            return

        new_folder_path = os.path.join(directory, folder_name)
        print(new_folder_path)
        print(f"Checking if folder exists: {new_folder_path}")  # Debugging line
        if os.path.exists(new_folder_path):
            QMessageBox.critical(self, "Error", "The action folder already exists.")
            return

        try:
            os.makedirs(new_folder_path)
            QMessageBox.information(self, "Success", "Action folder created successfully.")
            self.scan_directory(directory)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        
    def delete_folder(self):
        directory = self.directoryLineEdit.text()
        folder_name = self.action_comboBox.currentText()

        if not os.path.isdir(directory):
            QMessageBox.critical(self, "Error", "The specified directory does not exist. Check the chosen directory")
            return

        folder_path = os.path.join(directory, folder_name)
        if not os.path.exists(folder_path):
            QMessageBox.critical(self, "Error", "The folder does not exist.")
            return

        reply = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete the action folder '{folder_name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                os.rmdir(folder_path)
                QMessageBox.information(self, "Success", "Folder deleted successfully.")
                self.scan_directory(directory)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

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
            
    def center(self):
        # Get the screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        # Get the window geometry
        window_geometry = self.frameGeometry()
        # Move the window to the center of the screen
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def showActionsToTable(self):
        # Set the folder path
        folder_path = self.directoryLineEdit.text()

        # Get the list of folders and subfolders
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        # Set the number of rows and columns in the table
        self.action_table.setRowCount(len(folders))
        self.action_table.setColumnCount(2)

        # Populate the table with folder names and number of subfolders
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            subfolder_count = len([f for f in os.listdir(os.path.join(folder_path, folder)) if os.path.isdir(os.path.join(folder_path, folder, f))])
            subfolder_item = QTableWidgetItem(str(subfolder_count))

            self.action_table.setItem(row, 0, folder_item)
            self.action_table.setItem(row, 1, subfolder_item)

    def browse_button_functions(self):
        self.open_file_explorer()
        self.showActionsToTable()

    def startBlinking(self):
        self.status_label.setStyleSheet("""
                QLabel {
                    color: rgb(0, 255, 0);
                }
            """)
        self.status_label.setVisible(True) # Ensure the label is visible when starting
        self.blink_timer.start(750) # Start the blink timer with 750 milliseconds interval
    
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
        if self.hidden_tab_index == -1 or not self.MainTab.isTabVisible(self.hidden_tab_index):
            # Add the tab back
            self.MainTab.addTab(self.hidden_tab, self.hidden_tab_title)
            self.hidden_tab_index = self.MainTab.indexOf(self.hidden_tab)
            self.MainTab.setCurrentIndex(self.hidden_tab_index)
        else:
            # Remove the tab
            self.MainTab.removeTab(self.hidden_tab_index)
            self.hidden_tab_index = -1


