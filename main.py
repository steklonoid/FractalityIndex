import math
import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog
from mainWindow import UiMainWindow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import time
import random

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses, optimizers, metrics, activations, callbacks
from firstpart import Trade

class MainWindow(QMainWindow, UiMainWindow):

    version = '1.6.0'

    #   -----------------------------------------------------------
    def __init__(self):

        super().__init__()

        # создание визуальной формы
        self.setupui(self)
        self.show()
        self.filename = 'BTCUSDT-0521-0422-12month'
        # метапараметры:

    def loadbars_triggered(self):

        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
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

    def charcalc_triggered(self, f=1440):
        t1 = time.time()
        x = []
        f = 1440
        lnbars = np.log(self.bars[:, [2, 3, 4]]) # 0, 1, 2 // high, low, close
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
        df.to_csv('./data/' + self.filename + '_charcalc_' + str(f) + '.csv', index=False)

    def quantMAXOF_triggered(self, d=0.01):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "./data/", "CSV file (*.csv)")
        df = pd.read_csv(filename, dtype='float64')
        df_add = df['maxof']
        d = 0.007
        bins = [0, d, math.inf]
        t1 = time.time()

        a = pd.cut(df_add, bins=bins, labels=[0, 1])
        print(a.value_counts(normalize=True))

        df['maxof_class'] = pd.cut(df_add, bins=bins, labels=[0, 1])
        t = time.time() - t1
        print(t, df.shape, df.value_counts())
        df.to_csv('./data/' + self.filename + '_quantMAXOF_' + str(d) + '.csv', index=False)


    def quantfrac_triggered(self):
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
        intervals = np.array([5, 15, 60, 240, 1440])
        lastinterval = intervals[-1]
        interval_log = np.log(intervals)
        x = []
        for i in range(lastinterval, self.bars.shape[0]):
            list_bar = self.bars[i - lastinterval:i, [2, 3]]
            vj = [np.sum(np.subtract(np.amax(np.reshape(list_bar[:, [0]], (interval, -1)), axis=1),
                                     np.amin(np.reshape(list_bar[:, [1]], (interval, -1)), axis=1))) for
                  interval in intervals]
            vj_log = np.log(vj)
            (a, b, sigma) = mnk(interval_log, vj_log)
            x.append([self.bars[i, 0], a, b, sigma])

        t = time.time() - t1
        df = pd.DataFrame(x,
                          columns=['time', 'a', 'b', 'sigma'])
        print(t, df.shape)
        df.to_csv('./data/' + self.filename + '_quantfrac_' + str(lastinterval) + '.csv', index=False)

    def prepairNN_triggered(self):
        f1, _ = QFileDialog.getOpenFileName(self, "Open _quantMAXOF_ ", "./data/", "CSV file (*.csv)")
        f2, _ = QFileDialog.getOpenFileName(self, "Open _quantfrac_ ", "./data/", "CSV file (*.csv)")
        t1 = time.time()
        df1 = pd.read_csv(f1, dtype='float64')
        df2 = pd.read_csv(f2, dtype='float64')
        df = pd.merge(left=df1, right=df2, left_on='time', right_on='time')
        df = df[['time', 'a', 'b', 'sigma', 'maxof_class']]
        t = time.time() - t1
        print(t, df.shape, df.columns)
        df.to_csv('./data/' + self.filename + '_forNN_.csv', index=False)

    def trainNN_triggered(self):

        def splitwindows(past, future, x):
            n = x.shape[0] - past - future + 1
            xout = np.zeros((n, past, x.shape[1]))
            for i in range(n):
                xout[i] = x[i:i + past]
            return xout

        def model1():
            input_layer = layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))
            x = layers.Flatten()(input_layer)

            x = layers.Dense(units=20)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ELU()(x)

            x = layers.Dense(units=20)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ELU()(x)

            x = layers.Dense(units=20)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ELU()(x)

            x = layers.Dropout(rate=0.25)(x)

            output_layer = layers.Dense(units=1, activation='sigmoid')(x)
            model = models.Model(input_layer, output_layer)
            model.compile(optimizer=optimizers.RMSprop(),
                          loss=losses.BinaryCrossentropy(),
                          metrics=metrics.BinaryAccuracy())
            model.summary()
            return model

        def model2():
            input_layer = layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))
            x = layers.Conv1D(filters=64,
                              kernel_size=15,
                              strides=1,
                              padding='same',
                              activation='relu')(input_layer)
            x = layers.Conv1D(filters=64,
                              kernel_size=7,
                              strides=1,
                              padding='valid',
                              activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=3)(x)
            x = layers.Conv1D(filters=64,
                              kernel_size=5,
                              strides=1,
                              padding='valid',
                              activation='relu')(x)
            x = layers.GlobalMaxPooling1D()(x)
            output_layer = layers.Dense(units=1,
                             activation='sigmoid')(x)
            model = models.Model(input_layer, output_layer)
            model.compile(optimizer=optimizers.RMSprop(),
                          loss=losses.BinaryCrossentropy(),
                          metrics=metrics.BinaryAccuracy())
            model.summary()
            return model

        def model3():
            input_layer = layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))
            x = layers.Conv1D(filters=64,
                              kernel_size=15,
                              strides=1,
                              padding='same',
                              activation='relu')(input_layer)
            x = layers.MaxPooling1D(pool_size=3)(x)
            x = layers.Conv1D(filters=32,
                              kernel_size=7,
                              strides=1,
                              padding='valid',
                              activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=3)(x)
            x = layers.LSTM(32,
                           dropout=0.1,
                           recurrent_dropout=0.5)(x)
            output_layer = layers.Dense(units=1,
                                        activation='sigmoid')(x)
            model = models.Model(input_layer, output_layer)
            model.compile(optimizer=optimizers.RMSprop(),
                          loss=losses.BinaryCrossentropy(),
                          metrics=metrics.BinaryAccuracy())
            model.summary()
            return model


        t1 = time.time()
        df = pd.read_csv('./data/BTCUSDT-0521-0422-12month_forNN_.csv')
        x = df[['a', 'b', 'sigma']].astype(dtype='float64')
        y = df['maxof_class'].astype(dtype='int64')
        print(x.shape, y.shape)

        split_fraction = 0.50
        split_index = int(split_fraction * int(x.shape[0]))
        xtrain = x[:split_index]
        ytrain = y[:split_index]
        xtest = x[split_index:]
        ytest = y[split_index:]

        past = 60
        future = 1
        xtrain = splitwindows(past, future, xtrain)
        print(xtrain.shape)
        ytrain = ytrain[past:]
        print(ytrain.shape)
        xtest = splitwindows(past, future, xtest)
        print(xtest.shape)
        ytest = ytest[past:]
        print(ytest.shape)
        t = time.time() - t1
        print(t)
        # ++++++++++++++++++++++++++++++++++++++++++++
        # model = model1()
        model = model2()
        # ++++++++++++++++++++++++++++++++++++++++++++
        cb = [callbacks.TensorBoard(log_dir='tensor_log',
                                    histogram_freq=1,
                                    embeddings_freq=1)]
        history = model.fit(xtrain,
                            ytrain,
                            batch_size=512,
                            epochs=50,
                            validation_data=(xtest, ytest),
                            shuffle=False,
                            callbacks=cb)

        # print(history.history.keys())
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # acc_values = history.history['binary_accuracy']
        # val_acc_values = history.history['val_binary_accuracy']
        # epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.plot(epochs, acc_values, 'bo', label='Training acc')
        # plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        # plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()



    # ------------------------------------------------------------------------
        #   функция считает фрактальные коэффициенты на массиве баров bars
        #   для последних last баров
        #   с коэффициентом укрупнения koef
        #   так как изначально бары у нас минутные, то коэф. укрупнения означает расчет длля 5 минутных баров, если koef = 5 и т.д.
    def quantfrac_triggered(self, bars, shift, koef, intervals):

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

        lastinterval = intervals[-1]
        intervals_log = np.log(intervals)
        curbar_index = bars.shape[0] - shift - 1
        list_bar = bars[curbar_index - lastinterval * koef + 1:curbar_index + 1, [2, 3]]
        vj = [np.sum(np.subtract(np.amax(np.reshape(list_bar[:, [0]], (interval * koef, -1)), axis=1),
                                 np.amin(np.reshape(list_bar[:, [1]], (interval * koef, -1)), axis=1))) for
              interval in intervals]
        vj_log = np.log(vj)
        (a, b, sigma) = mnk(intervals_log, vj_log)
        x = [vj_log, a, b, sigma]
        return x

    def loadbars_2_triggered(self):
        #   читает файлы, полученные с https://data.binance.vision/
        #   формат файлов: CSV - файлы, значения разделенные запятыми,
        #   значения полей слева направо:
        #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
        listcol = [0, 1, 2, 3, 4, 5]  # Time / Open / High / Low / Close / Volume
        filename = 'data/BTCUSDT-1m-2022-04.csv'
        self.bars = np.genfromtxt(filename, delimiter=',')[:, listcol]
        self.bars[:, 0] //= 1000
        self.graphicsView.bararray = np.flip(self.bars, axis=0)
        self.graphicsView.repaint()

    def star_animation(self):
        koefs = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
        koefs_logs = np.log(koefs)
        intervals = np.array([1, 2, 4, 8, 16, 32])
        intervals_log = np.log(intervals)
        x, y = np.meshgrid(intervals_log, koefs_logs)
        z = np.zeros((koefs.shape[0], intervals.shape[0]))
        count_frames = 200
        fr = np.zeros((count_frames, koefs.shape[0], intervals.shape[0]))

        t1 = time.time()
        for shift in range(count_frames):
            for index, koef in enumerate(koefs):
                res = self.quantfrac_triggered(self.bars, shift, koef, intervals)
                z[index] = res[0]
            fr[shift] = z
            t = self.bars[-1-shift, 0]
        t = time.time() - t1
        print(t)
        print(fr.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        def animate(i):
            ax.clear()
            ax.plot_surface(x, y, fr[i], cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)
            ax.set_zlim(5, 15)
            self.graphicsView.currentbartime = self.bars[-1-i,0]
            self.graphicsView.repaint()

        self.ani = animation.FuncAnimation(fig, animate, frames=range(count_frames), interval=100)
        plt.show()
    # ------------------------------------------------------------------------


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



app = QApplication([])
win = MainWindow()
sys.exit(app.exec())
