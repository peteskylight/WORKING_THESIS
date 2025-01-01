import sys
import cv2
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QTabWidget, QWidget, QAction, QMenuBar, QMenu, QStatusBar
from PySide2.QtCore import QRect, QCoreApplication, QMetaObject
from PySide2.QtGui import QFont

from pygrabber.dshow_graph import FilterGraph

from utils import CameraFeed
from gui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera_feed_instance = CameraFeed(self.camera_feed)
        self.openCamera.clicked.connect(self.start_camera)
        self.populate_camera_combo_box()

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