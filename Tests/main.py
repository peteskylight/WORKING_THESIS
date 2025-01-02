import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QCheckBox, QLabel
from checkbox_checker import CheckboxChecker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.checkbox = QCheckBox("Check me", self)
        self.checkbox.move(50, 50)
        self.checkbox.stateChanged.connect(self.update_label)

        self.label = QLabel("nah", self)
        self.label.move(50, 100)

        self.checkbox_checker = CheckboxChecker(self.checkbox, self.label)

    def update_label(self):
        if self.checkbox.isChecked():
            self.label.setText("Checked")
        else:
            self.label.setText("nah")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
