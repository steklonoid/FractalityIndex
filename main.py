import sys
from PyQt6.QtWidgets import QMainWindow, QApplication
from mainWindow import UiMainWindow
import matplotlib as plt
import numpy as np
import time
import random

import tensorflow as tf
from tensorflow import keras
from keras import layers


class Trade():
    def __init__(self, id=00,  time=0, qty = 0, price=0, side = '', open_qty = 0, dictclosedtrades = dict({}), profit = 0):
        self.id = id
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

    def sampleNN(self):
        num_classes = 10
        input_shape = (28, 28, 1)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.summary()
        batch_size = 128
        epochs = 15

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def pb_start_clicked(self):

        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
        self.filename = 'BTCUSDT-1m-2022-04'
        listcol = [0, 1, 2, 3, 4, 5]        # Time / Open / High / Low / Close / Volume
        bars01 = np.genfromtxt('./data/BTCUSDT-1m-2022-01.csv', delimiter=',')[:, listcol]
        bars02 = np.genfromtxt('./data/BTCUSDT-1m-2022-02.csv', delimiter=',')[:, listcol]
        bars03 = np.genfromtxt('./data/BTCUSDT-1m-2022-03.csv', delimiter=',')[:, listcol]
        bars04 = np.genfromtxt('./data/' + self.filename + '.csv', delimiter=',')[:, listcol]
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

        def calculateprofit():
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
                        dct = dict(trade.dictclosedtrades)
                        dct.update({pos: closed_qty})
                        trade.dictclosedtrades = dct
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
                dictclosedtrades = {x: x.open_qty for x in listinpos}
                koef = 1 if sidepos == 'BUY' else -1
                profit = sum([(x.open_qty * koef * (price - x.price)) for x in listinpos])
                trades.append(Trade(id=id,
                                    time=time,
                                    qty=qty,
                                    price=price,
                                    side=side,
                                    dictclosedtrades=dictclosedtrades,
                                    profit=profit))

        random.seed(10)
        #   создаем словарь торговых сделок
        trades = []
        id = 0
        #   случайный приказ
        for i in range(self.bars.shape[0]):
            a = random.randint(0, 1)
            b = random.random()
            if b < 0.001:   #   частота случайного приказа
                trades.append(Trade(id=id,
                                    time=self.bars[i, 0],
                                    qty=1,
                                    price=self.bars[i, 1],
                                    side='BUY' if a == 0 else 'SELL'))
                id += 1

        calculateprofit()

        print(len(trades))
        profit_list = [trade.profit for trade in trades]
        print(profit_list)
        print(sum(profit_list))

    def frackoef_calculate(self, f):

        def mnk(xlist, ylist):
            n = np.size(xlist)
            sx = np.sum(xlist)
            sy = np.sum(ylist)
            sxy = np.sum(np.multiply(xlist, ylist))
            sxx = np.sum(np.square(xlist))
            a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            b = (sy - a * sx) / n
            sigma = np.sum(np.square(np.subtract(ylist, a * xlist + b)))
            return (a, b, sigma)

        frackoef_list = []
        intervals = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        interval_log = np.log(intervals)
        for i in range(intervals[-1], self.bars.shape[0] - f):
            list_bar = self.bars[i - lastinterval:i, [1, 2]]
            vj = [np.sum(np.subtract(np.amax(np.reshape(list_bar[:, [1]], (interval, -1)), axis=1),
                                     np.amin(np.reshape(list_bar[:, [0]], (interval, -1)), axis=1))) for
                  interval in cf_intervals]
            vj_log = np.log(vj)
            (a, b, sigma) = mnk(interval_log, vj_log)
            frackoef_list.append([np_listbar[i, 0], a, b, sigma])

    def zapoln(self):
        pass
        # sumvi = np.sum(self.bars[i:i + f, 2], axis=0) - np.sum(self.bars[i:i + f, 3], axis=0)
        # fill = sumvi / (f * (xmax - xmin))

    def pb_work_clicked(self):

        def mnk(xlist, ylist):
            n = np.size(xlist)
            sx = np.sum(xlist)
            sy = np.sum(ylist)
            sxy = np.sum(np.multiply(xlist, ylist))
            sxx = np.sum(np.square(xlist))
            a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            b = (sy - a * sx) / n
            sigma = np.sum(np.square(np.subtract(ylist, a * xlist + b)))
            return (a, b, sigma)

        t1 = time.time()
        f = int(self.cb_period.currentText())
        intervals = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        lastinterval = intervals[-1]
        interval_log = np.log(intervals)
        x = []
        for i in range(lastinterval, self.bars.shape[0] - f):
            hmax = np.amax(self.bars[i:i+f, 2], axis=0)
            lmin = np.amin(self.bars[i:i + f, 3], axis=0)

            list_bar = self.bars[i - lastinterval:i, [2, 3]]
            vj = [np.sum(np.subtract(np.amax(np.reshape(list_bar[:, [0]], (interval, -1)), axis=1),
                                     np.amin(np.reshape(list_bar[:, [1]], (interval, -1)), axis=1))) for
                  interval in intervals]
            vj_log = np.log(vj)
            (a, b, sigma) = mnk(interval_log, vj_log)
            x.append([self.bars[i, 0], self.bars[i, 1], hmax, lmin, a, b, sigma])

        xnp = np.array(x, dtype=float)
        t = time.time() - t1
        print(t, ' : рассчитали hmax, lmin' )
        np.savetxt('./data/' + self.filename + '_' + str(f) + '_forwardmaxmin.csv',
                   xnp,
                   fmt=['%.0f' ,'%.2f' ,'%.2f', '%.2f', '%.10f' ,'%.10f', '%.10f'],
                   header='time, open, hmax, lmin, a, b, sigma',
                   delimiter=',')



app = QApplication([])
win = MainWindow()
sys.exit(app.exec())
