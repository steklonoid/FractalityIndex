from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QComboBox, QMenuBar
from PyQt6.QtGui import QAction
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
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.centralwidget.setLayout(self.gridLayout)

        self.graphicsView = DisplayField()
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 5, 1)

        self.menubar = QMenuBar()
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        mainwindow.setMenuBar(self.menubar)

        self.filemenu = self.menubar.addMenu("Операции")
        self.loadbars = QAction("Загрузка баров из файлов", self)
        self.loadbars.triggered.connect(mainwindow.loadbars_triggered)
        self.filemenu.addAction(self.loadbars)
        self.charcalc = QAction("Расчет характеристик", self)
        self.charcalc.triggered.connect(mainwindow.charcalc_triggered)
        self.filemenu.addAction(self.charcalc)
        self.quantcalc = QAction("Расчет квантования MAXOF", self)
        self.quantcalc.triggered.connect(mainwindow.quantMAXOF_triggered)
        self.filemenu.addAction(self.quantcalc)
        self.quantfrac = QAction("Расчет фрактальных a, b, sigma", self)
        self.quantfrac.triggered.connect(mainwindow.quantfrac_triggered)
        self.filemenu.addAction(self.quantfrac)
        self.trainNN = QAction("Тренировка нейросети", self)
        self.trainNN.triggered.connect(mainwindow.trainNN_triggered)
        self.filemenu.addAction(self.trainNN)



        # self.pb_start = QPushButton()
        # self.pb_start.setText('START')
        # self.pb_start.clicked.connect(mainwindow.pb_start_clicked)
        # self.gridLayout.addWidget(self.pb_start, 0, 1, 1, 1)
        #
        # self.pb_frac = QPushButton()
        # self.pb_frac.setText('Расчет фракталов')
        # self.pb_frac.clicked.connect(mainwindow.pb_frac_clicked)
        # self.gridLayout.addWidget(self.pb_frac, 1, 1, 1, 1)
        #
        # self.pb_profit = QPushButton()
        # self.pb_profit.setText('Расчет прибыли')
        # self.pb_profit.clicked.connect(mainwindow.pb_profit_clicked)
        # self.gridLayout.addWidget(self.pb_profit, 1, 2, 1, 1)
        #
        # self.cb_period = QComboBox()
        # self.cb_period.addItems(['60', '240', '1440'])
        # self.gridLayout.addWidget(self.cb_period, 2, 1, 1, 1)
        #
        # self.pb_work = QPushButton()
        # self.pb_work.setText('Работы разные')
        # self.pb_work.clicked.connect(mainwindow.pb_work_clicked)
        # self.gridLayout.addWidget(self.pb_work, 3, 1, 1, 1)
        #
        # self.pb_nn = QPushButton()
        # self.pb_nn.setText('Нейросеть')
        # self.pb_nn.clicked.connect(mainwindow.pb_nn_clicked)
        # self.gridLayout.addWidget(self.pb_nn, 3, 2, 1, 1)



