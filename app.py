from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from window import MainWindow
import sys

app = QApplication(sys.argv)
window = MainWindow()
window.show() 

app.exec()