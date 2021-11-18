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
                if mode == 1:
                    if s[i, 1] == max(s[i - 2, 1], s[i - 1, 1], s[i, 1], s[i + 1, 1], s[i + 2, 1]):
                        fl = True
                else:
                    if s[i, 1] == min(s[i - 2, 1], s[i - 1, 1], s[i, 1], s[i + 1, 1], s[i + 2, 1]):
                        fl = True
                if fl:
                    s_out.append([s[i, 0], s[i, 1], sumvol])
                    sumvol = 0
            return np.array(s_out)

        bars1 = np.genfromtxt('./data/BTCUSDT-1m-2021-01.csv', delimiter=',')[:,[0, 2, 3, 5]]
        bars2 = np.genfromtxt('./data/BTCUSDT-1m-2021-02.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars3 = np.genfromtxt('./data/BTCUSDT-1m-2021-03.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars4 = np.genfromtxt('./data/BTCUSDT-1m-2021-04.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars5 = np.genfromtxt('./data/BTCUSDT-1m-2021-05.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars6 = np.genfromtxt('./data/BTCUSDT-1m-2021-06.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars7 = np.genfromtxt('./data/BTCUSDT-1m-2021-07.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars8 = np.genfromtxt('./data/BTCUSDT-1m-2021-08.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars9 = np.genfromtxt('./data/BTCUSDT-1m-2021-09.csv', delimiter=',')[:, [0, 2, 3, 5]]
        bars10 = np.genfromtxt('./data/BTCUSDT-1m-2021-10.csv', delimiter=',')[:, [0, 2, 3, 5]]

        # bars [time, high, low, volume]
        bars = np.concatenate((bars1, bars2, bars3, bars4, bars5, bars6, bars7, bars8, bars9, bars10))
        print(bars.shape)

        f_set = {}
        f_rank = 0
        f_up = bars[:, [0, 1, 3]]
        f_down = bars[:, [0, 2, 3]]
        while f_up.shape[0] > 4 or f_down.shape[0] > 4:
            f_up = calc_fractals(1, f_up)
            f_down = calc_fractals(0, f_down)
            f_rank += 1
            f_set[f_rank] = {'f_up': f_up, 'f_down': f_down}

        print(f_set)

        # bars_re = np.reshape(bars, (-1, 10))
        # print(bars_re.shape)


app = QApplication([])
win = MainWindow()
sys.exit(app.exec_())
