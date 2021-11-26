import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSlot
from mainWindow import UiMainWindow
import pandas as pd
import numpy as np

class MainWindow(QMainWindow, UiMainWindow):

    version = '1.5.0'

    #   -----------------------------------------------------------
    def __init__(self):

        super().__init__()

        # создание визуальной формы
        self.setupui(self)
        self.show()
    #   ---------------------------------------------------------------------

    def pb_start_clicked(self):

        def calc_fractals(mode, s):
            s_out = []
            len_s = s.shape[0]
            sumvol = s[0, 2] + s[1, 2]
            for i in range(2, len_s - 3):
                sumvol += s[i, 2]
                fl = False
                if mode == 1 and s[i, 1] > s[i - 2, 1] and  s[i, 1] > s[i - 1, 1] and s[i, 1] > s[i + 1, 1] and s[i, 1] > s[i + 2, 1]:
                        fl = True
                elif mode == 0 and s[i, 1] < s[i - 2, 1] and  s[i, 1] < s[i - 1, 1] and s[i, 1] < s[i + 1, 1] and s[i, 1] < s[i + 2, 1]:
                        fl = True
                if fl:
                    s_out.append([s[i, 0], s[i, 1], sumvol])
                    sumvol = 0
            return np.array(s_out)

        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore

        listcol = [0, 1, 2, 3, 4, 5]        # Time / Open / High / Low / Close / Volume
        bars01 = np.genfromtxt('./data/BTCUSDT-1m-2021-01.csv', delimiter=',')[:, listcol]
        bars02 = np.genfromtxt('./data/BTCUSDT-1m-2021-02.csv', delimiter=',')[:, listcol]
        bars03 = np.genfromtxt('./data/BTCUSDT-1m-2021-03.csv', delimiter=',')[:, listcol]
        bars04 = np.genfromtxt('./data/BTCUSDT-1m-2021-04.csv', delimiter=',')[:, listcol]
        bars05 = np.genfromtxt('./data/BTCUSDT-1m-2021-05.csv', delimiter=',')[:, listcol]
        bars06 = np.genfromtxt('./data/BTCUSDT-1m-2021-06.csv', delimiter=',')[:, listcol]
        bars07 = np.genfromtxt('./data/BTCUSDT-1m-2021-07.csv', delimiter=',')[:, listcol]
        bars08 = np.genfromtxt('./data/BTCUSDT-1m-2021-08.csv', delimiter=',')[:, listcol]
        bars09 = np.genfromtxt('./data/BTCUSDT-1m-2021-09.csv', delimiter=',')[:, listcol]
        bars10 = np.genfromtxt('./data/BTCUSDT-1m-2021-10.csv', delimiter=',')[:, listcol]

        bars = np.concatenate((bars01, bars02, bars03, bars04, bars05, bars06, bars07, bars08, bars09, bars10))
        bars[:, 0] //= 1000
        self.graphicsView.bararray = np.flip(bars, axis=0)
        self.graphicsView.repaint()

        f_set = {}
        f_rank = 0
        f_up = bars[:, [0, 2, 5]]
        f_down = bars[:, [0, 3, 5]]
        while f_up.shape[0] > 4 or f_down.shape[0] > 4:
            f_up = calc_fractals(1, f_up)
            f_down = calc_fractals(0, f_down)
            f_rank += 1
            f_set[f_rank] = {'f_up': f_up, 'f_down': f_down}

        self.graphicsView.fractaluparray = np.flip(np.array(f_set[3]['f_up']), axis=0)
        self.graphicsView.fractaldownarray = np.flip(np.array(f_set[3]['f_down']), axis=0)
        self.graphicsView.repaint()

        print({k:[len(v['f_up']), len(v['f_down'])] for k, v in f_set.items()})
        # print(f_set[16])
        # print(f_set[17])
        # print(f_set[18])
        # bars_re = np.reshape(bars, (-1, 10))
        # print(bars_re.shape)


app = QApplication([])
win = MainWindow()
sys.exit(app.exec_())
