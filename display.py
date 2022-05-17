import math
from PyQt6.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QPixmap, QMouseEvent, QWheelEvent, QFont, QColor
from PyQt6.QtWidgets import QWidget, QSizePolicy
import numpy as np
from datetime import datetime


class DisplayField(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.bararray = np.zeros((0, 6))    #   t / o / h / l / c / vol
        self.fractaluparray = np.zeros((0, 3))
        self.fractaldownarray = np.zeros((0, 3))
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

        self.down_bar_pen = QPen(Qt.GlobalColor.darkYellow)
        self.down_bar_brush = QBrush(Qt.GlobalColor.darkYellow)
        self.up_bar_pen = QPen(Qt.GlobalColor.darkYellow)
        self.up_bar_brush = QBrush(Qt.GlobalColor.black)
        self.fractaluppen = QPen(Qt.GlobalColor.green)
        self.fractaldownpen = QPen(Qt.GlobalColor.red)
        self.trade_line_pen = QPen(Qt.GlobalColor.white)
        self.trade_line_pen.setWidth(3)
        self.trade_line_pen.setStyle(Qt.PenStyle.DashLine)
        self.open_buy_position = QPixmap()
        self.open_buy_position.load("./up_arrow_short_small.png")
        self.open_sell_position = QPixmap()
        self.open_sell_position.load("./down_arrow_short_small.png")
        self.close_plus_position = QPen(Qt.GlobalColor.green)
        self.close_plus_position.setWidth(5)
        self.close_minus_position = QPen(Qt.GlobalColor.red)
        self.close_minus_position.setWidth(5)
        self.fontinfo = QFont("Helvetica", 10, QFont.Weight.Bold)
        self.fontaxe = QFont("Helvetica", 8, QFont.Weight.Normal)
        self.axes_pen = QPen(QColor(128, 128, 128))
        self.axes_pen.setWidth(2)
        self.grid_pen = QPen(QColor(128, 128, 128))
        self.grid_pen.setStyle(Qt.PenStyle.DotLine)

    def clearView(self, painter):
        self.width = painter.viewport().width()  # текущая ширина окна рисования
        self.height = painter.viewport().height()  # текущая высота окна рисования
        painter.fillRect(0, 0, self.width, self.height, Qt.GlobalColor.black)  # очищаем окно (черный цвет)

    def drawAxes(self, painter):
        painter.setPen(self.axes_pen)
        painter.drawLine(0, self.upaxe, self.width, self.upaxe)
        painter.drawLine(self.width - self.rightshiftx, 0, self.width - self.rightshiftx, self.height)
        painter.drawLine(0, self.height - self.downaxe, self.width, self.height - self.downaxe)

    def paintEvent(self, event):
        painter = QPainter(self)

        self.clearView(painter)
        self.drawAxes(painter)


        if self.bararray.shape[0] > 0:
            self.barshift = math.floor(self.timeshift / self.timeframe)
            self.barshift = min(self.barshift, self.bararray.shape[0] - self.barcount - 3)
            self.scalex = (self.width - self.rightshiftx) / (self.barcount * self.timeframe)    # масштаб по оХ
            barwidth = self.scalex * self.timeframe
            a = self.bararray[self.barshift:self.barshift + self.barcount + 1]

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
                    painter.fillRect(QRectF(ax[i], aclose[i], barwidth, aopen[i] - aclose[i]), self.down_bar_brush)
                    painter.drawRect(QRectF(ax[i], aclose[i], barwidth, aopen[i] - aclose[i]))
                    painter.drawLine(QLineF(centerx, aclose[i], centerx, ahigh[i]))
                else:
                    painter.setPen(self.up_bar_pen)
                    painter.drawLine(QLineF(centerx, alow[i], centerx, aclose[i]))
                    painter.fillRect(QRectF(ax[i], aopen[i], barwidth, aclose[i] - aopen[i]), self.up_bar_brush)
                    painter.drawRect(QRectF(ax[i], aopen[i], barwidth, aclose[i] - aopen[i]))
                    painter.drawLine(QLineF(centerx, aopen[i], centerx, ahigh[i]))

            #   отрисовка фракталов
            painter.setPen(self.fractaluppen)
            a = self.fractaluparray[self.fractaluparray[:, 0] > lefttime]
            ax = np.multiply(a[:, 0], self.scalex) + bx
            ay = np.multiply(a[:, 1], -self.scaley) + by
            for i in range(0, a.shape[0]):
                centerx = ax[i] + barwidth / 2
                painter.drawEllipse(QPointF(centerx, ay[i]), 5, 5)
            painter.setPen(self.fractaldownpen)
            a = self.fractaldownarray[self.fractaldownarray[:, 0] > lefttime]
            ax = np.multiply(a[:, 0], self.scalex) + bx
            ay = np.multiply(a[:, 1], -self.scaley) + by
            for i in range(0, a.shape[0]):
                centerx = ax[i] + barwidth / 2
                painter.drawEllipse(QPointF(centerx, ay[i]), 5, 5)

            #   отрисока оси y
            minpx = 50
            m = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            curm = m[0]
            for mi in m:
                if mi * self.scaley > minpx:
                    curm = mi
                    break
            minris = curm * math.ceil(miny / curm)
            masris = np.arange(minris, maxy, curm)
            masrisy = np.multiply(masris, -self.scaley) + by
            painter.setPen(self.grid_pen)
            painter.setFont(self.fontaxe)
            for i in range(masrisy.shape[0]):
                painter.drawLine(QLineF(0, masrisy[i], self.width - self.rightshiftx, masrisy[i]))
                painter.drawText(QPointF(self.width - self.rightshiftx + 5, masrisy[i]), str(int(masris[i])))

            # #   отрисовка оси x
            # minpx = 200
            # m = [60, 300, 600, 1800, 3600, 7200, 14400, 43200, 86400]
            # dx = self.barcount * self.timeframe
            # sx = dx / (self.width - self.rightshiftx)
            # curm = m[0]
            # for mi in m:
            #     if mi / sx > minpx:
            #         curm = mi
            #         break
            # maxris = curm * math.ceil(a[-1, 0] / curm)
            # masris = np.arange(maxris - dx, maxris + self.timeframe, curm)
            # masrisx = np.multiply(masris, self.scalex) + bx
            # painter.setPen(self.grid_pen)
            # painter.setFont(self.fontaxe)
            # for i in range(masrisx.shape[0]):
            #     painter.drawLine(masrisx[i], self.upaxe, masrisx[i], self.height - self.downaxe)
            #     painter.drawText(masrisx[i], self.height - 5, str(datetime.utcfromtimestamp(masris[i])))

        # painter.setFont(self.fontinfo)
        # painter.setPen(Qt.white)
        # s = str(round(self.balance, 3)) + " / " + str(self.tradescount)
        # painter.drawText(10, 15, s)

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        if self.bararray.shape[0] > 0:
            if self.mousemovexpos != 0:
                delta = a0.pos().x() - self.mousemovexpos
                self.timeshift += delta / self.scalex
                self.timeshift = max(0, self.timeshift)
            self.mousemovexpos = a0.pos().x()
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