import sys
import os
import numpy as np
import psutil
import GPUtil
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PySide2.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime
from PySide2.QtGui import QFont

from pygrabber.dshow_graph import FilterGraph

from utils.camera import CameraFeed
from utils.drawing_utils import DrawingUtils
from gui import Ui_MainWindow
from trackers import PoseDetection


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.isRecording = False

        self.camera_feed_instance = CameraFeed(self.camera_feed, self.white_frame_feed, self)
        self.drawing_utils = DrawingUtils()
        
        self.closeCamera.clicked.connect(self.camera_feed_instance.stop_camera)
        
        self.openCamera.clicked.connect(self.start_camera)
        
        self.browseButton.clicked.connect(self.open_file_explorer)
        
        self.refresh_button.clicked.connect(lambda: self.scan_directory(self.directoryLineEdit.text()))
        
        self.add_action_button.clicked.connect(self.add_folder)
        
        self.delete_action_button.clicked.connect(self.delete_folder)
        
        self.recording_button.clicked.connect(self.toggle_button)
        
        self.populate_camera_combo_box()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_usage)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
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