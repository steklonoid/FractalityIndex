
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QComboBox, QVBoxLayout, QSplitter, QSizePolicy, QTableWidget, QHeaderView
from display import DisplayField


class UiMainWindow(object):

    def __init__(self):
        self.buttonlist = []
        self.numcontbuttonlist = []

    def setupui(self, mainwindow):
        mainwindow.setObjectName("MainWindow")
        mainwindow.setWindowTitle('DLM Bot v ' + mainwindow.version)
        mainwindow.resize(1000, 600)

        self.centralwidget = QWidget(mainwindow)
        mainwindow.setCentralWidget(self.centralwidget)

        self.gridLayout = QGridLayout()
        self.gridLayout.setContentsMargins(30, 30, 30, 30)
        self.centralwidget.setLayout(self.gridLayout)

        self.graphicsView = DisplayField()
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)

        self.pb_start = QPushButton()
        self.pb_start.setText('START')
        self.pb_start.clicked.connect(mainwindow.pb_start_clicked)
        self.gridLayout.addWidget(self.pb_start, 0, 1, 1, 1)


