import sys
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt, QTimer)
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import *

from gui import (Ui_MainWindow, MainWindow)

if __name__ == "__main__":
    # Check if a QApplication instance already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        print("QApplication instance already exists.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
        