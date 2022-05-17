import sys
from PyQt6.QtWidgets import QMainWindow, QApplication
from mainWindow import UiMainWindow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random

import tensorflow as tf
from tensorflow import keras
from keras import layers, models


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

    version = '1.6.0'

    #   -----------------------------------------------------------
    def __init__(self):

        super().__init__()

        # создание визуальной формы
        self.setupui(self)
        self.show()

        # метапараметры:


    def pb_nn_clicked(self):
        # number;time;open;hmax;lmin;MAXOPEN;MINOPEN;a;b;sigma;ADD;ADD Номер интервала;DIFF;DIFF Номер интервала;MAXOF;MAXOF Номер интервала;DIFFMAXOF;DIFFMAXOF Номер интервала
        preliminarydata = np.genfromtxt('./data/Quant.csv', delimiter=';', dtype='float32')
        print('Входные данные: ', preliminarydata.shape)

        prex = preliminarydata[:, 7]
        prey = preliminarydata[:, 15]

        past = 120
        future = 1
        learning_rate = 0.001
        n = preliminarydata.shape[0] - past - future + 1
        x = np.zeros((n, past))
        y = prey[past - future + 1:]
        y = keras.utils.to_categorical(y)
        for i in range(n):
            x[i] = prex[i:i + past]
        print(x.shape, y.shape)

        split_fraction = 0.80
        train_split = int(split_fraction * int(x.shape[0]))
        x_train = x[:train_split]
        x_mean = np.mean(x_train)
        x_train -= x_mean
        x_std = np.std(x_train)
        x_train /= x_std
        y_train = y[:train_split]
        # y_mean = np.mean(y_train)
        # y_train -= y_mean
        # y_std = np.std(y_train)
        # y_train /= y_std
        x_test = x[train_split:]
        x_test -= x_mean
        x_test /= x_std
        y_test = y[train_split:]
        # y_test -= y_mean
        # y_test /= y_std

        model = models.Sequential()
        model.add(layers.Dense(27, input_shape=(x_train.shape[1],), activation='relu'))
        model.add(layers.Dense(18, activation='relu'))
        model.add(layers.Dense(9, activation='softmax'))
        model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        model.summary()

        history = model.fit(
            x_train,
            y_train,
            epochs=50,
            validation_data=(x_test, y_test),
            batch_size=2048)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def _pb_nn_clicked(self):

        def slider_window(data, past, future):
            number_of_slices = data.shape[0] - past - future + 1
            newdata = np.zeros((past, data.shape[1], number_of_slices))
            for i in range(number_of_slices):
                newdata[:, :, i] = data[i:i + past, :]
            return newdata

        # number;time;open;hmax;lmin;MAXOPEN;MINOPEN;a;b;sigma;ADD;ADD Номер интервала;DIFF;DIFF Номер интервала;MAXOF;MAXOF Номер интервала;DIFFMAXOF;DIFFMAXOF Номер интервала
        preliminarydata = np.genfromtxt('./data/Quant.csv', delimiter=';', dtype='float32')
        print('Входные данные: ', preliminarydata.shape)

        #   разделение множества на тренировочное и обучающее
        train_fraction = 0.8 # процент обучающего множества в исходных данных
        num_train = int(train_fraction * int(preliminarydata.shape[0]))
        values = preliminarydata[:, [7]]
        labels = keras.utils.to_categorical(preliminarydata[:, [15]])
        print('Данные и метки: ', values.shape, labels.shape)

        train_values = values[:num_train]
        train_labels = labels[:num_train]
        test_values = values[num_train:]
        test_labels = labels[num_train:]
        print('Тренировочное и проверочное множества: ',train_values.shape, test_values.shape)

        #   нормализация и подготовка данных
        # mean = train_values.mean(axis=0)
        # train_values -= mean
        # std = train_values.std(axis=0)
        # train_values /= std
        # test_values -= mean
        # test_values /= std

        #   подготовка датасетов

        past = 1440
        future = 1
        batch_size = 1

        train_values = slider_window(train_values, past, future)
        train_labels = train_labels[past:]
        test_values = slider_window(test_values, past, future)
        test_labels = test_labels[past:]

        print(train_values.shape, train_labels.shape, test_values.shape, test_labels.shape)

          # создание модели
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(past, values.shape[1])))
        # model.add(layers.LSTM(32))
        model.add(layers.Dense(labels.shape[1], activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        #
        #   тренировка модели
        history = model.fit(
            train_values,
            train_labels,
            epochs=5,
            validation_data=(test_values, test_labels),
            batch_size=batch_size)




    def pb_start_clicked(self):

        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
        self.filename = 'BTCUSDT-0521-0422-12month'
        listcol = [0, 1, 2, 3, 4, 5]        # Time / Open / High / Low / Close / Volume
        t1 = time.time()
        bars05 = np.genfromtxt('./data/BTCUSDT-1m-2021-05.csv', delimiter=',')[:, listcol]
        bars06 = np.genfromtxt('./data/BTCUSDT-1m-2021-06.csv', delimiter=',')[:, listcol]
        bars07 = np.genfromtxt('./data/BTCUSDT-1m-2021-07.csv', delimiter=',')[:, listcol]
        bars08 = np.genfromtxt('./data/BTCUSDT-1m-2021-08.csv', delimiter=',')[:, listcol]
        bars09 = np.genfromtxt('./data/BTCUSDT-1m-2021-09.csv', delimiter=',')[:, listcol]
        bars10 = np.genfromtxt('./data/BTCUSDT-1m-2021-10.csv', delimiter=',')[:, listcol]
        bars11 = np.genfromtxt('./data/BTCUSDT-1m-2021-11.csv', delimiter=',')[:, listcol]
        bars12 = np.genfromtxt('./data/BTCUSDT-1m-2021-12.csv', delimiter=',')[:, listcol]
        bars01 = np.genfromtxt('./data/BTCUSDT-1m-2022-01.csv', delimiter=',')[:, listcol]
        bars02 = np.genfromtxt('./data/BTCUSDT-1m-2022-02.csv', delimiter=',')[:, listcol]
        bars03 = np.genfromtxt('./data/BTCUSDT-1m-2022-03.csv', delimiter=',')[:, listcol]
        bars04 = np.genfromtxt('./data/BTCUSDT-1m-2022-04.csv', delimiter=',')[:, listcol]


        bars = np.concatenate((bars05, bars06, bars07, bars08, bars09, bars10, bars11, bars12, bars01, bars02, bars03, bars04, ))
        self.bars = bars
        self.bars[:, 0] //= 1000
        self.graphicsView.bararray = np.flip(self.bars, axis=0)
        self.graphicsView.repaint()
        t = time.time() - t1
        print(t, self.bars.shape)

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

    def zapoln(self):
        pass
        # sumvi = np.sum(self.bars[i:i + f, 2], axis=0) - np.sum(self.bars[i:i + f, 3], axis=0)
        # fill = sumvi / (f * (xmax - xmin))

    def charcalc_triggered(self):
        t1 = time.time()
        f = int(self.cb_period.currentText())
        x = []
        lnbars = np.log(self.bars[:, [2, 3, 4]]) # high, low, close
        for i in range(self.bars.shape[0] - f):
            hmax = np.amax(lnbars[i:i + f, 0], axis=0)
            lmin = np.amin(lnbars[i:i + f, 1], axis=0)
            maxclose = hmax - lnbars[i, 2]
            minclose = lmin - lnbars[i, 2]
            add = maxclose + minclose
            diff = maxclose - minclose
            maxof = max(maxclose, -minclose)
            diffmaxof = diff - maxof
            x.append([self.bars[i, 0], lnbars[i, 2], hmax, lmin, maxclose, minclose, add, diff, maxof, diffmaxof])

        t = time.time() - t1
        df = pd.DataFrame(x,
                          columns=['time', 'close', 'hmax', 'lmin', 'maxclose', 'minclose', 'add', 'diff', 'maxof', 'diffmaxof'])
        print(t, df.shape)
        df.to_csv('./data/' + self.filename + '_charcalc_' + str(f) + '.csv')



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
