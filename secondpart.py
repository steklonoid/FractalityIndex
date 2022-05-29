import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
import time
import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QWidget, QGridLayout

class UiMainWindow(object):
    def setupui(self, mainwindow):
        mainwindow.setObjectName("MainWindow")
        mainwindow.setWindowTitle('Animation')
        mainwindow.resize(1000, 600)

        self.centralwidget = QWidget(mainwindow)
        mainwindow.setCentralWidget(self.centralwidget)

        self.gridLayout = QGridLayout()
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.centralwidget.setLayout(self.gridLayout)

class MainWindow(QMainWindow, UiMainWindow):
    def __init__(self):

        super().__init__()

        # создание визуальной формы
        self.setupui(self)
        self.show()

app = QApplication([])
win = MainWindow()
sys.exit(app.exec())