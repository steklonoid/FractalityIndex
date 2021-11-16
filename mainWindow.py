from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QLabel, QComboBox, QVBoxLayout, QSplitter, QSizePolicy, QTableWidget, QHeaderView
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap, QMouseEvent, QWheelEvent, QFont, QColor
import numpy as np
import math
from datetime import datetime


class DisplayField(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.bararray = np.zeros((0, 5))    #   t / o / h / l / c
        self.tradearray = np.zeros((0, 6))  #   opentime / openpx / closetime / closepx / side / profit

        self.rightshiftx = 40
        self.downaxe = 20
        self.upaxe = 20

        self.barcount = 100  # кол-во отображаемых баров, шт
        self.timeframe = 60
        self.timeshift = 0
        self.barshift = 0

        self.scalex = 1
        self.scaley = 1

        self.mousemovexpos = 0
        self.width = 0
        self.height = 0

        self.balance = 0
        self.tradescount = 0

        self.down_bar_pen = QPen(Qt.darkYellow)
        self.down_bar_brush = QBrush(Qt.darkYellow)
        self.up_bar_pen = QPen(Qt.darkYellow)
        self.up_bar_brush = QBrush(Qt.black)
        self.trade_line_pen = QPen(Qt.white)
        self.trade_line_pen.setWidth(3)
        self.trade_line_pen.setStyle(Qt.DashLine)
        self.open_buy_position = QPixmap()
        self.open_buy_position.load("./up_arrow_short_small.png")
        self.open_sell_position = QPixmap()
        self.open_sell_position.load("./down_arrow_short_small.png")
        self.close_plus_position = QPen(Qt.green)
        self.close_plus_position.setWidth(5)
        self.close_minus_position = QPen(Qt.red)
        self.close_minus_position.setWidth(5)
        self.fontinfo = QFont("Helvetica", 10, QFont.Bold)
        self.fontaxe = QFont("Helvetica", 8, QFont.Normal)
        self.axes_pen = QPen(QColor(128, 128, 128))
        self.axes_pen.setWidth(2)
        self.grid_pen = QPen(QColor(128, 128, 128))
        self.grid_pen.setStyle(Qt.DotLine)

    def paintEvent(self, event):
        painter = QPainter(self)

        self.width = painter.viewport().width()  # текущая ширина окна рисования
        self.height = painter.viewport().height()  # текущая высота окна рисования
        painter.fillRect(0, 0, self.width, self.height, Qt.black)  # очищаем окно (черный цвет)
        painter.setPen(self.axes_pen)
        painter.drawLine(0, self.upaxe, self.width, self.upaxe)
        painter.drawLine(self.width - self.rightshiftx, 0, self.width - self.rightshiftx, self.height)
        painter.drawLine(0, self.height - self.downaxe, self.width, self.height - self.downaxe)

        if self.bararray.shape[0] > 0:
            self.barshift = math.floor(self.timeshift / self.timeframe)
            self.barshift = min(self.barshift, self.bararray.shape[0] - self.barcount - 3)
            self.scalex = (self.width - self.rightshiftx) / (self.barcount * self.timeframe)    # масштаб по оХ
            barwidth = self.scalex * self.timeframe
            a = self.bararray[self.barshift:self.barcount + self.barshift + 1]

            miny = np.amin(a[:, 3], axis=0)
            maxy = np.amax(a[:, 2], axis=0)
            self.scaley = (self.height - self.downaxe - self.upaxe) / (maxy - miny)  # масштаб по оУ

            righttime = self.bararray[0, 0] + self.timeframe - self.timeshift
            lefttime = righttime -  self.barcount * self.timeframe

            bx = self.width - self.rightshiftx - righttime * self.scalex
            by = miny * self.scaley + self.height - self.downaxe  # сдвиг по оУ
            # расчет баров
            ax = np.multiply(a[:, 0], self.scalex) + bx
            ahigh = np.multiply(a[:, 2], -self.scaley) + by
            alow = np.multiply(a[:, 3], -self.scaley) + by
            aopen =  np.multiply(a[:, 1], -self.scaley) + by
            aclose = np.multiply(a[:, 4], -self.scaley) + by
            # отрисовка баров
            for i in range(0, a.shape[0]):
                centerx = ax[i] + barwidth / 2
                if aclose[i] < aopen[i]: #  down bar
                    painter.setPen(self.down_bar_pen)
                    painter.drawLine(QLineF(centerx, alow[i], centerx, aopen[i]))
                    painter.fillRect(ax[i], aclose[i], barwidth, aopen[i] - aclose[i], self.down_bar_brush)
                    painter.drawRect(ax[i], aclose[i], barwidth, aopen[i] - aclose[i])
                    painter.drawLine(QLineF(centerx, aclose[i], centerx, ahigh[i]))
                else:
                    painter.setPen(self.up_bar_pen)
                    painter.drawLine(QLineF(centerx, alow[i], centerx, aclose[i]))
                    painter.fillRect(ax[i], aopen[i], barwidth, aclose[i] - aopen[i], self.up_bar_brush)
                    painter.drawRect(ax[i], aopen[i], barwidth, aclose[i] - aopen[i])
                    painter.drawLine(QLineF(centerx, aopen[i], centerx, ahigh[i]))
            # расчет и отрисока торговых сделок
            a = np.array([v for v in self.tradearray if (v[0] >= lefttime and v[0] <= righttime) or (v[2] >= lefttime and v[2] <= righttime)])
            if a.shape[0] > 0:
                x1 = np.multiply(a[:, 0], self.scalex) + bx
                y1 = np.multiply(a[:, 1], -self.scaley) + by
                x2 = np.multiply(a[:, 2], self.scalex) + bx
                y2 = np.multiply(a[:, 3], -self.scaley) + by

                for i in range(0, a.shape[0]):
                    painter.setPen(self.trade_line_pen)
                    painter.drawLine(x1[i], y1[i], x2[i], y2[i])
                    barwidth = max(self.timeframe * self.scalex, 20)
                    if a[i, 4] == 0:
                        painter.drawPixmap(int(x1[i] - barwidth / 2), int(y1[i] - barwidth), barwidth, barwidth, self.open_buy_position)
                    else:
                        painter.drawPixmap(int(x1[i] - barwidth / 2), int(y1[i]), barwidth, barwidth, self.open_sell_position)
                    if a[i, 5] >= 0:
                        painter.setPen(self.close_plus_position)
                        painter.drawEllipse(QPointF(x2[i], y2[i]), barwidth // 2, barwidth // 2)
                    else:
                        painter.setPen(self.close_minus_position)
                        painter.drawEllipse(QPointF(x2[i], y2[i]), barwidth // 2, barwidth // 2)
            #   отрисока оси y
            minpx = 50
            m = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            dy = maxy - miny
            sy = dy / (self.height - self.downaxe - self.upaxe)
            curm = m[0]
            for mi in m:
                if mi / sy > minpx:
                    curm = mi
                    break
            minris = curm * math.ceil(miny / curm)
            masris = np.arange(minris, maxy, curm)
            masrisy = np.multiply(masris, -self.scaley) + by
            painter.setPen(self.grid_pen)
            painter.setFont(self.fontaxe)
            for i in range(masrisy.shape[0]):
                painter.drawLine(0, masrisy[i], self.width - self.rightshiftx, masrisy[i])
                painter.drawText(self.width - self.rightshiftx + 5, masrisy[i], str(int(masris[i])))

            #   отрисовка оси x
            minpx = 200
            m = [60, 300, 600, 1800, 3600, 7200, 14400, 43200, 86400]
            dx = self.barcount * self.timeframe
            sx = dx / (self.width - self.rightshiftx)
            curm = m[0]
            for mi in m:
                if mi / sx > minpx:
                    curm = mi
                    break
            maxris = curm * math.ceil(a[-1, 0] / curm)
            masris = np.arange(maxris - dx, maxris + self.timeframe, curm)
            masrisx = np.multiply(masris, self.scalex) + bx
            painter.setPen(self.grid_pen)
            painter.setFont(self.fontaxe)
            for i in range(masrisx.shape[0]):
                painter.drawLine(masrisx[i], self.upaxe, masrisx[i], self.height - self.downaxe)
                painter.drawText(masrisx[i], self.height - 5, str(datetime.utcfromtimestamp(masris[i])))

        painter.setFont(self.fontinfo)
        painter.setPen(Qt.white)
        s = str(round(self.balance, 3)) + " / " + str(self.tradescount)
        painter.drawText(10, 15, s)

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        if self.bararray.shape[0] > 0:
            if self.mousemovexpos != 0:
                delta = a0.x() - self.mousemovexpos
                self.timeshift += delta / self.scalex
                self.timeshift = max(0, self.timeshift)
            self.mousemovexpos = a0.x()
            self.repaint()

    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        self.mousemovexpos = 0

    def wheelEvent(self, a0: QWheelEvent) -> None:
        if self.bararray.shape[0] > 0:
            if a0.angleDelta().y() > 0:
                barcount = self.barcount * 1.2
                barwidth = (self.width - self.rightshiftx) // barcount
                if barwidth >= 1:
                    self.barcount = round(barcount)
            else:
                barcount = self.barcount / 1.2
                if barcount > 10:
                    self.barcount = round(barcount)
            self.repaint()

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

        self.hsplitter = QSplitter(Qt.Horizontal)
        self.gridLayout.addWidget(self.hsplitter, 0, 0, 1, 1)

        self.vbox = QWidget()
        self.vboxlayout = QVBoxLayout()
        self.vbox.setLayout(self.vboxlayout)
        self.hsplitter.addWidget(self.vbox)

        self.l_conagents = QLabel()
        self.l_conagents.setText('Агент')
        self.l_conagents.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.vboxlayout.addWidget(self.l_conagents)

        self.cb_conagents = QComboBox()
        self.cb_conagents.setToolTip('Выбор агента')
        for k in mainwindow.conagents.keys():
            self.cb_conagents.addItem(k)
        self.cb_conagents.currentIndexChanged.connect(self.cb_conagents_currentIndexChanged)
        self.vboxlayout.addWidget(self.cb_conagents)

        self.l_par_conagents = QLabel()
        self.l_par_conagents.setText('Параметры агента')
        self.l_par_conagents.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.vboxlayout.addWidget(self.l_par_conagents)

        self.par_conagents = QTableWidget()
        self.par_conagents.setObjectName('par_conagents')
        self.par_conagents.setColumnCount(2)
        self.par_conagents.setHorizontalHeaderLabels(['Параметр', 'Значение'])
        self.par_conagents.verticalHeader().hide()
        self.par_conagents.horizontalScrollBar().hide()
        self.par_conagents.setColumnWidth(0, 100)
        self.par_conagents.setColumnWidth(1, 150)
        self.par_conagents.setContextMenuPolicy(Qt.CustomContextMenu)
        self.par_conagents.customContextMenuRequested.connect(self.par_conagents_customContextMenuRequested)
        self.vboxlayout.addWidget(self.par_conagents)
        self.cb_conagents_currentIndexChanged()

        self.l_strategy = QLabel()
        self.l_strategy.setText('Стратегия')
        self.l_strategy.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.vboxlayout.addWidget(self.l_strategy)

        self.cb_strategy = QComboBox()
        self.cb_strategy.setToolTip('Выбор стратегии')
        for k in mainwindow.strategies.keys():
            self.cb_strategy.addItem(k)
        self.cb_strategy.currentIndexChanged.connect(self.cb_strategy_currentIndexChanged)
        self.vboxlayout.addWidget(self.cb_strategy)

        self.l_par_strategy = QLabel()
        self.l_par_strategy.setText('Параметры стратегии')
        self.l_par_strategy.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.vboxlayout.addWidget(self.l_par_strategy)

        self.par_strategy = QTableWidget()
        self.par_strategy.setObjectName('par_conagents')
        self.par_strategy.setColumnCount(2)
        self.par_strategy.setHorizontalHeaderLabels(['Параметр', 'Значение'])
        self.par_strategy.verticalHeader().hide()
        self.par_strategy.horizontalScrollBar().hide()
        self.par_strategy.setColumnWidth(0, 100)
        self.par_strategy.setColumnWidth(1, 150)
        self.vboxlayout.addWidget(self.par_strategy)
        self.cb_strategy_currentIndexChanged()

        self.pb_start = QPushButton()
        self.pb_start.setText('Старт')
        self.pb_start.clicked.connect(self.pb_start_clicked)
        self.vboxlayout.addWidget(self.pb_start)

        self.pb_placeorder = QPushButton()
        self.pb_placeorder.setText('Разместить ордер')
        self.pb_placeorder.clicked.connect(self.pb_placeorder_clicked)
        self.vboxlayout.addWidget(self.pb_placeorder)

        self.pb_cancelorder = QPushButton()
        self.pb_cancelorder.setText('Старт')
        self.pb_cancelorder.clicked.connect(self.pb_cancelorder_clicked)
        self.vboxlayout.addWidget(self.pb_cancelorder)

        self.l_info = QLabel()
        self.l_info.setAlignment(Qt.AlignCenter)
        self.vboxlayout.addWidget(self.l_info)

        self.pb_showgraf = QPushButton()
        self.pb_showgraf.setText('Показать график')
        self.pb_showgraf.clicked.connect(self.pb_showgraf_clicked)
        self.vboxlayout.addWidget(self.pb_showgraf)

        self.vsplitter = QSplitter(Qt.Vertical)
        self.vsplitter.setObjectName('vsplitter')
        self.graphicsView = DisplayField()
        self.vsplitter.addWidget(self.graphicsView)
        self.hsplitter.addWidget(self.vsplitter)
        self.hsplitter.setSizes([300, 600])

        cssFile = "./mainWindow.css"
        with open(cssFile, "r") as fh:
            self.setStyleSheet(fh.read())


