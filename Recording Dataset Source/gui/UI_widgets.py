import sys
import cv2
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QTabWidget, QWidget, QAction, QMenuBar, QMenu, QStatusBar
from PySide2.QtCore import QRect, QCoreApplication, QMetaObject
from PySide2.QtGui import QFont

from utils import CameraFeed


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1091, 750)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionExport = QAction(MainWindow)
        self.actionExport.setObjectName(u"actionExport")
        self.actionExit_2 = QAction(MainWindow)
        self.actionExit_2.setObjectName(u"actionExit_2")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.MainTab = QTabWidget(self.centralwidget)
        self.MainTab.setObjectName(u"MainTab")
        self.MainTab.setGeometry(QRect(0, 0, 1080, 720))
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(18)
        self.MainTab.setFont(font)
        self.analyticstTab = QWidget()
        self.analyticstTab.setObjectName(u"analyticstTab")
        self.MainTab.addTab(self.analyticstTab, "")
        self.createDatasetTab = QWidget()
        self.createDatasetTab.setObjectName(u"createDatasetTab")
        self.cameraComboBox = QComboBox(self.createDatasetTab)
        self.cameraComboBox.setObjectName(u"cameraComboBox")
        self.cameraComboBox.setGeometry(QRect(80, 10, 171, 31))
        font1 = QFont()
        font1.setPointSize(14)
        self.cameraComboBox.setFont(font1)
        self.label_Camera = QLabel(self.createDatasetTab)
        self.label_Camera.setObjectName(u"label_Camera")
        self.label_Camera.setGeometry(QRect(10, 10, 81, 31))
        self.label_Camera.setFont(font1)
        self.camera_feed = QLabel(self.createDatasetTab)
        self.camera_feed.setObjectName(u"camera_feed")
        self.camera_feed.setGeometry(QRect(20, 60, 611, 421))
        self.openCamera = QPushButton(self.createDatasetTab)
        self.openCamera.setObjectName(u"openCamera")
        self.openCamera.setGeometry(QRect(260, 10, 131, 31))
        
        font2 = QFont()
        font2.setPointSize(12)
        self.openCamera.setFont(font2)
        self.MainTab.addTab(self.createDatasetTab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1091, 21))
        self.menuARMIS = QMenu(self.menubar)
        self.menuARMIS.setObjectName(u"menuARMIS")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuARMIS.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menuARMIS.addAction(self.actionExit)
        self.menuARMIS.addAction(self.actionOpen)
        self.menuARMIS.addAction(self.actionSave)
        self.menuARMIS.addAction(self.actionExport)
        self.menuARMIS.addAction(self.actionExit_2)

        self.retranslateUi(MainWindow)

        self.MainTab.setCurrentIndex(1)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"New", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionExport.setText(QCoreApplication.translate("MainWindow", u"Export", None))
        self.actionExit_2.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.MainTab.setTabText(self.MainTab.indexOf(self.analyticstTab), QCoreApplication.translate("MainWindow", u"Analytics", None))
        self.label_Camera.setText(QCoreApplication.translate("MainWindow", u"Camera", None))
        self.camera_feed.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.openCamera.setText(QCoreApplication.translate("MainWindow", u"Open Camera", None))
        self.MainTab.setTabText(self.MainTab.indexOf(self.createDatasetTab), QCoreApplication.translate("MainWindow", u"Create Dataset", None))
        self.menuARMIS.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", u"View", None))
    # retranslateUi