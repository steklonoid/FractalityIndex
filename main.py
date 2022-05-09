import sys
from PyQt6.QtWidgets import QMainWindow, QApplication
from mainWindow import UiMainWindow
import matplotlib as plt
import numpy as np
import time
import random


class Trade():
    def __init__(self, time=0, qty = 0, price=0, side = '', open_qty = 0, dictclosedtrades = {}, profit = 0):
        self.time = time
        self.qty = qty
        self.price = price
        self.side = side

        self.open_qty = open_qty
        self.dictclosedtrades = dictclosedtrades
        self.profit = profit

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

        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore

        listcol = [0, 1, 2, 3, 4, 5]        # Time / Open / High / Low / Close / Volume
        bars01 = np.genfromtxt('./data/BTCUSDT-1m-2022-01.csv', delimiter=',')[:, listcol]
        bars02 = np.genfromtxt('./data/BTCUSDT-1m-2022-02.csv', delimiter=',')[:, listcol]
        bars03 = np.genfromtxt('./data/BTCUSDT-1m-2022-03.csv', delimiter=',')[:, listcol]
        bars04 = np.genfromtxt('./data/BTCUSDT-1m-2022-04.csv', delimiter=',')[:, listcol]
        # bars05 = np.genfromtxt('./data/BTCUSDT-1m-2021-05.csv', delimiter=',')[:, listcol]
        # bars06 = np.genfromtxt('./data/BTCUSDT-1m-2021-06.csv', delimiter=',')[:, listcol]
        # bars07 = np.genfromtxt('./data/BTCUSDT-1m-2021-07.csv', delimiter=',')[:, listcol]
        # bars08 = np.genfromtxt('./data/BTCUSDT-1m-2021-08.csv', delimiter=',')[:, listcol]
        # bars09 = np.genfromtxt('./data/BTCUSDT-1m-2021-09.csv', delimiter=',')[:, listcol]
        # bars10 = np.genfromtxt('./data/BTCUSDT-1m-2021-10.csv', delimiter=',')[:, listcol]

        # bars = np.concatenate((bars01, bars02, bars03, bars04, bars05, bars06, bars07, bars08, bars09, bars10))
        self.bars = np.concatenate((bars01, bars02, bars03, bars04))
        self.bars[:, 0] //= 1000
        print(self.bars.shape)
        self.graphicsView.bararray = np.flip(self.bars, axis=0)
        self.graphicsView.repaint()

    def pb_frac_clicked(self):

        def calc_fractals(mode, s):
            s_out = []
            len_s = s.shape[0]
            sumvol = s[0, 2] + s[1, 2]
            for i in range(2, len_s - 3):
                sumvol += s[i, 2]
                fl = False
                if mode == 1:
                    if s[i, 1] > s[i - 2, 1] and s[i, 1] > s[i - 1, 1] and s[i, 1] > s[i + 1, 1] and s[i, 1] > s[i + 2, 1]:
                        fl = True
                else:
                    if s[i, 1] < s[i - 2, 1] and  s[i, 1] < s[i - 1, 1] and s[i, 1] < s[i + 1, 1] and s[i, 1] < s[i + 2, 1]:
                        fl = True
                if fl:
                    s_out.append([s[i, 0], s[i, 1], sumvol])
                    sumvol = 0
            return np.array(s_out)


        t1 = time.time()
        f_set = {}
        f_rank = 0
        f_up = self.bars[:, [0, 2, 5]]
        f_down = self.bars[:, [0, 3, 5]]
        while f_up.shape[0] > 4 or f_down.shape[0] > 4:
            f_up = calc_fractals(1, f_up)
            f_down = calc_fractals(0, f_down)
            f_rank += 1
            f_set[f_rank] = {'f_up': f_up, 'f_down': f_down}

        t = time.time() - t1
        print(t)
        self.graphicsView.fractaluparray = np.flip(np.array(f_set[2]['f_up']), axis=0)
        self.graphicsView.fractaldownarray = np.flip(np.array(f_set[2]['f_down']), axis=0)
        self.graphicsView.repaint()
        print({k: [len(v['f_up']), len(v['f_down'])] for k, v in f_set.items()})

    def pb_profit_clicked(self):
        random.seed(10)
        #   создаем словарь торговых сделок
        trades = []
        #   случайный приказ
        for i in range(self.bars.shape[0]):
            a = random.randint(0, 1)
            b = random.random()
            if b < 0.001:   #   частота случайного приказа
                trades.append(Trade(time=self.bars[i, 0],
                                    qty=1,
                                    price=self.bars[i, 1],
                                    side='BUY' if a == 0 else 'SELL'))

        #   рассчет прибыли по сделкам
        listinpos = []
        sidepos = ''
        for trade in trades:
            _qty = trade.qty
            if len(listinpos) == 0 or sidepos == trade.side:
                sidepos = trade.side
                trade.open_qty = _qty
                listinpos.append(trade)
            else:
                for pos in listinpos:
                    closed_qty = min(_qty, pos.open_qty)
                    _qty -= closed_qty
                    pos.open_qty -= closed_qty

                    koef = 1 if sidepos == 'BUY' else -1
                    profit = closed_qty * koef * (trade.price - pos.price)
                    trade.dictclosedtrades.update({closed_qty:pos})
                    trade.profit += profit

                    if _qty == 0:
                        break

                listinpos = [x for x in listinpos if x.open_qty > 0]
                if _qty > 0:
                    sidepos = trade.side
                    trade.open_qty = _qty
                    listinpos.append(trade)

        if len(listinpos) > 0:
            time = self.bars[-1, 0]
            qty = sum([x.open_qty for x in listinpos])
            price = self.bars[-1, 1]
            side = 'BUY' if sidepos == 'SELL' else 'SELL'
            dictclosedtrades = {x.open_qty:x for x in listinpos}
            koef = 1 if sidepos == 'BUY' else -1
            profit = sum([(x.open_qty * koef * (price - x.price)) for x in listinpos])
            trades.append(Trade(time=time,
                                qty=qty,
                                price=price,
                                side=side,
                                dictclosedtrades=dictclosedtrades,
                                profit=profit))

        print(len(trades))








app = QApplication([])
win = MainWindow()
sys.exit(app.exec())
