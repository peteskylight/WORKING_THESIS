# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'RECORDING-UI2QoXzYc.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFrame,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QWidget)

class Ui_RecordingWindow(object):
    def setupUi(self, RecordingWindow):
        if not RecordingWindow.objectName():
            RecordingWindow.setObjectName(u"RecordingWindow")
        RecordingWindow.resize(1023, 574)
        self.center_view_recording_label = QFrame(RecordingWindow)
        self.center_view_recording_label.setObjectName(u"center_view_recording_label")
        self.center_view_recording_label.setGeometry(QRect(520, 80, 491, 301))
        self.center_view_recording_label.setFrameShape(QFrame.Shape.WinPanel)
        self.center_view_recording_label.setFrameShadow(QFrame.Shadow.Raised)
        self.center_view_recording_label.setLineWidth(2)
        self.label_3 = QLabel(self.center_view_recording_label)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 0, 121, 31))
        font = QFont()
        font.setPointSize(14)
        font.setItalic(True)
        self.label_3.setFont(font)
        self.front_view_recording_label = QFrame(RecordingWindow)
        self.front_view_recording_label.setObjectName(u"front_view_recording_label")
        self.front_view_recording_label.setGeometry(QRect(20, 80, 491, 301))
        self.front_view_recording_label.setFrameShape(QFrame.Shape.WinPanel)
        self.front_view_recording_label.setFrameShadow(QFrame.Shadow.Raised)
        self.front_view_recording_label.setLineWidth(2)
        self.label_4 = QLabel(self.front_view_recording_label)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(0, 0, 111, 31))
        self.label_4.setFont(font)
        self.stop_recording_videos_button = QPushButton(RecordingWindow)
        self.stop_recording_videos_button.setObjectName(u"stop_recording_videos_button")
        self.stop_recording_videos_button.setGeometry(QRect(270, 510, 241, 51))
        font1 = QFont()
        font1.setPointSize(18)
        self.stop_recording_videos_button.setFont(font1)
        self.pushButton_5 = QPushButton(RecordingWindow)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(920, 10, 81, 51))
        font2 = QFont()
        font2.setPointSize(16)
        self.pushButton_5.setFont(font2)
        self.label_5 = QLabel(RecordingWindow)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 10, 1011, 61))
        font3 = QFont()
        font3.setFamilies([u"Sitka"])
        font3.setPointSize(36)
        self.label_5.setFont(font3)
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.front_recording_camera_combo_box = QComboBox(RecordingWindow)
        self.front_recording_camera_combo_box.addItem("")
        self.front_recording_camera_combo_box.addItem("")
        self.front_recording_camera_combo_box.setObjectName(u"front_recording_camera_combo_box")
        self.front_recording_camera_combo_box.setGeometry(QRect(20, 390, 301, 41))
        font4 = QFont()
        font4.setPointSize(14)
        self.front_recording_camera_combo_box.setFont(font4)
        self.center_recording_camera_combo_box = QComboBox(RecordingWindow)
        self.center_recording_camera_combo_box.addItem("")
        self.center_recording_camera_combo_box.addItem("")
        self.center_recording_camera_combo_box.setObjectName(u"center_recording_camera_combo_box")
        self.center_recording_camera_combo_box.setGeometry(QRect(520, 390, 301, 41))
        self.center_recording_camera_combo_box.setFont(font4)
        self.label_6 = QLabel(RecordingWindow)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(20, 450, 91, 41))
        self.label_6.setFont(font2)
        self.label_7 = QLabel(RecordingWindow)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(460, 450, 49, 41))
        self.label_7.setFont(font2)
        self.open_center_recording_camera = QPushButton(RecordingWindow)
        self.open_center_recording_camera.setObjectName(u"open_center_recording_camera")
        self.open_center_recording_camera.setGeometry(QRect(830, 390, 181, 41))
        self.open_center_recording_camera.setFont(font2)
        self.open_front_recording_camera = QPushButton(RecordingWindow)
        self.open_front_recording_camera.setObjectName(u"open_front_recording_camera")
        self.open_front_recording_camera.setGeometry(QRect(330, 390, 181, 41))
        self.open_front_recording_camera.setFont(font2)
        self.fileName_line_edit = QLineEdit(RecordingWindow)
        self.fileName_line_edit.setObjectName(u"fileName_line_edit")
        self.fileName_line_edit.setGeometry(QRect(120, 450, 331, 41))
        self.fileName_line_edit.setFont(font2)
        self.fileName_line_edit.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.start_recording_videos_button = QPushButton(RecordingWindow)
        self.start_recording_videos_button.setObjectName(u"start_recording_videos_button")
        self.start_recording_videos_button.setGeometry(QRect(20, 510, 241, 51))
        self.start_recording_videos_button.setFont(font1)
        self.import_recording_button = QPushButton(RecordingWindow)
        self.import_recording_button.setObjectName(u"import_recording_button")
        self.import_recording_button.setGeometry(QRect(670, 440, 211, 121))
        font5 = QFont()
        font5.setPointSize(20)
        self.import_recording_button.setFont(font5)

        self.retranslateUi(RecordingWindow)

        QMetaObject.connectSlotsByName(RecordingWindow)
    # setupUi

    def retranslateUi(self, RecordingWindow):
        RecordingWindow.setWindowTitle(QCoreApplication.translate("RecordingWindow", u"Dialog", None))
        self.label_3.setText(QCoreApplication.translate("RecordingWindow", u"  Center View", None))
        self.label_4.setText(QCoreApplication.translate("RecordingWindow", u"  Front View", None))
        self.stop_recording_videos_button.setText(QCoreApplication.translate("RecordingWindow", u"STOP RECORDING", None))
        self.pushButton_5.setText(QCoreApplication.translate("RecordingWindow", u"Exit", None))
        self.label_5.setText(QCoreApplication.translate("RecordingWindow", u"RECORDING PANELS", None))
        self.front_recording_camera_combo_box.setItemText(0, QCoreApplication.translate("RecordingWindow", u"FRONT VIEW CAMERA", None))
        self.front_recording_camera_combo_box.setItemText(1, QCoreApplication.translate("RecordingWindow", u"CENTER VIEW CAMERA", None))

        self.center_recording_camera_combo_box.setItemText(0, QCoreApplication.translate("RecordingWindow", u"FRONT VIEW CAMERA", None))
        self.center_recording_camera_combo_box.setItemText(1, QCoreApplication.translate("RecordingWindow", u"CENTER VIEW CAMERA", None))

        self.label_6.setText(QCoreApplication.translate("RecordingWindow", u"Filename:", None))
        self.label_7.setText(QCoreApplication.translate("RecordingWindow", u".MP4", None))
        self.open_center_recording_camera.setText(QCoreApplication.translate("RecordingWindow", u"Open Camera", None))
        self.open_front_recording_camera.setText(QCoreApplication.translate("RecordingWindow", u"Open Camera", None))
        self.fileName_line_edit.setText("")
        self.start_recording_videos_button.setText(QCoreApplication.translate("RecordingWindow", u"START RECORDING", None))
        self.import_recording_button.setText(QCoreApplication.translate("RecordingWindow", u"IMPORT\n"
"RECORDINGS", None))
    # retranslateUi

