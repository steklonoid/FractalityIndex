import sys
import queue
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem, QMenu, QFileDialog
from PyQt5.QtCore import QSettings, pyqtSlot, Qt
from PyQt5.QtGui import QCursor
from mainWindow import UiMainWindow

from agent_localfile import LocalFile
from agent_binance import BinanceClient
from agent_dgtx import DGTXClient
from agent_poloniex import PoloniexClient

from strategy_random_01 import Random_01
from strategy_stohastic_01 import Stohastic_01
from strategy_bigspred_01 import BigSpred
from strategy_writetofile_01 import WriteToFile_01

from base import FromQToF

from threading import Lock

import numpy as np


ex = {'BTCUSD-PERP':{'TICK_SIZE':5, 'TICK_VALUE':0.1},'ETHUSD-PERP':{'TICK_SIZE':1, 'TICK_VALUE':1}}



class MainWindow(QMainWindow, UiMainWindow):

    settings = QSettings("./config.ini", QSettings.IniFormat)
    version = '1.5.0'
    lock = Lock()

    #   -----------------------------------------------------------
    def __init__(self):

        super().__init__()
        # список агентов - формируется выпадающий список
        self.conagents = {'Digitex':DGTXClient, 'Poloniex': PoloniexClient, 'Binance': BinanceClient, 'File': LocalFile}
        # список стратегий - формируется выпадающий список
        self.strategies = {'Random': Random_01, 'Stohastic 3 уровня': Stohastic_01, 'Большой спред': BigSpred, 'Write to file': WriteToFile_01}
        # -----------------------------------------------------------------------
        # создаем очередь для получения сообщений от агента
        self.agentq = queue.Queue()
        # создаем поток, обрабатывающий очередь и посылающий сообщения из очереди в функцию
        self.agentreceiver = FromQToF(self.agentq, self.receivefromagent)
        self.agentreceiver.daemon = True
        self.agentreceiver.start()
        # создание визуальной формы
        self.setupui(self)
        self.show()
        #   ---------------------------------------------------------------------

    def receivefromagent(self, data):
        if data:
            msg_type = data.get('msg_type')
            msg = data.get('msg')
            if msg_type == 'tick_price':
                self.strategy.tick_price(msg)
            elif msg_type == 'order_book':
                self.strategy.order_book(msg)
            elif msg_type == 'order_book_updated':
                self.strategy.order_book_updated(msg)
            elif msg_type == 'order_placed':
                self.strategy.order_placed(msg)
            elif msg_type == 'order_canceled':
                self.strategy.order_canceled(msg)
            else:
                pass
        else:
            self.stat()

    def stat(self):
        self.l_info.setText('Рассчет окончен')
        self.graphicsView.balance = self.strategy.parameters['balance']
        self.graphicsView.tradescount = len(self.strategy.closedcontractlist)
        self.graphicsView.bararray = np.array([[i, self.strategy.bardict[i]['o'], self.strategy.bardict[i]['h'], self.strategy.bardict[i]['l'], self.strategy.bardict[i]['c']] for i in sorted(self.strategy.bardict.keys(), reverse=True)])
        sides = {'BUY':0, 'SELL':1}
        self.graphicsView.tradearray = np.array([[v.opentime, v.openpx, v.closetime, v.closepx, sides[v.side], v.profit] for v in self.strategy.closedcontractlist.values()])
        self.graphicsView.repaint()

    @pyqtSlot()
    def cb_conagents_currentIndexChanged(self):
        self.agent = self.conagents[self.cb_conagents.currentText()](self.agentq)
        self.agent.daemon = True
        self.par_conagents.setRowCount(0)
        par = self.agent.parameters
        rownumber = 0
        for k, v in par.items():
            self.par_conagents.setRowCount(self.par_conagents.rowCount() + 1)
            item = QTableWidgetItem()
            item.setText(k)
            self.par_conagents.setItem(rownumber, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, v)
            self.par_conagents.setItem(rownumber, 1, item)
            rownumber += 1


    @pyqtSlot()
    def cb_strategy_currentIndexChanged(self):
        self.strategy = self.strategies[self.cb_strategy.currentText()]()
        self.par_strategy.setRowCount(0)
        par = self.strategy.parameters
        rownumber = 0
        for k, v in par.items():
            self.par_strategy.setRowCount(self.par_strategy.rowCount() + 1)
            item = QTableWidgetItem()
            item.setText(str(k))
            self.par_strategy.setItem(rownumber, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, v)
            self.par_strategy.setItem(rownumber, 1, item)
            rownumber += 1

    @pyqtSlot()
    def customContextMenuTriggered(self, num):
        if num == 1:
            fname = QFileDialog.getOpenFileName(self, "Выберите файл базы", "", "*.csv file (*.csv)")[0]
            indexes = self.par_conagents.selectedIndexes()
            row = indexes[0].row()
            self.par_conagents.item(row, 1).setData(Qt.DisplayRole, fname)

    @pyqtSlot()
    def par_conagents_customContextMenuRequested(self):
        indexes = self.par_conagents.selectedIndexes()
        if indexes:
            row = indexes[0].row()
            parname = self.par_conagents.item(row, 0).data(Qt.DisplayRole)
            if parname == 'filename':
                menu = QMenu()
                menu.addAction('Выбрать файл').triggered.connect(lambda: self.customContextMenuTriggered(1))
                menu.exec_(QCursor.pos())

    @pyqtSlot()
    def pb_placeorder_clicked(self):
        self.agent.po()

    @pyqtSlot()
    def pb_cancelorder_clicked(self):
        self.agent.co()

    @pyqtSlot()
    def pb_start_clicked(self):
        self.agent = self.conagents[self.cb_conagents.currentText()](self.agentq)
        self.agent.daemon = True
        for i in range(self.par_conagents.rowCount()):
            self.agent.parameters[self.par_conagents.item(i, 0).text()] = self.par_conagents.item(i, 1).data(Qt.DisplayRole)

        self.strategy = self.strategies[self.cb_strategy.currentText()]()
        self.strategy.agent = self.agent
        for i in range(self.par_strategy.rowCount()):
            self.strategy.parameters[self.par_strategy.item(i, 0).text()] = self.par_strategy.item(i, 1).data(Qt.DisplayRole)

        self.l_info.setText('Идет рассчет')
        self.agent.start()

        # filename = 'BTCUSDT-1m-2021-05.csv'
        # ar = np.genfromtxt(filename, delimiter=',')
        # ar[:, 0] -= ar[0, 0]
        #
        # step = 20
        # X = np.arange(5)
        # Y = np.arange(5)
        # Z = np.zeros((X.shape[0], Y.shape[0]))
        # for x in X:
        #     for y in Y:
        #         takeprofit = step * (x + 1)
        #         stoploss = step * (y + 1)
        #
        #
        #         self.strategy.parameters = {'balance': 1000,
        #                               'koef':1,
        #                               'qty': 1,
        #                               'sumstoptrading': -1000000,
        #                               'takeprofit': 60,
        #                               'stoploss': 60}
        #
        #         for i in range(ar.shape[0]):
        #             self.strategy.message_index(ar[i])
        #         print(takeprofit, stoploss, self.strategy.parameters['balance'])
        #         Z[x, y] = self.strategy.parameters['balance']
        #
        # print(Z)
        # indx = np.unravel_index(Z.argmax(), Z.shape)
        # print(indx)
        # print((indx[0] + 1) * step, (indx[1] + 1) * step)
        # print(np.amax(Z))
        #
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(1, 2, 1, projection='3d')
        # X, Y = np.meshgrid(X, Y)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=10)
        # plt.show()

    @pyqtSlot()
    def pb_showgraf_clicked(self):
        # fig, axes = plt.subplots(1, 1)
        # axes[0].plot(self.strategy.balancehistory.keys(), self.strategy.balancehistory.values())
        plt.plot(self.strategy.balancehistory.keys(), self.strategy.balancehistory.values())
        plt.show()


app = QApplication([])
win = MainWindow()
sys.exit(app.exec_())
