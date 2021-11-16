import numpy as np
from base import Strategy, Contract, ClosedContract


class Stohastic_01(Strategy):
    # Описание:
    # Работаем с синей основной линией стохастика с параметрами 14 1 3
    # Как только линия касается снизу уровня 20 - покупаем
    # при касании линией уровня 50 снизу - закрываем половину
    # при касании линией уровня 20 сверху - закрываем все
    # при касании линией уровня 80 снизу - закрываем все
    # Для продажи симметрично



    def __init__(self, agent=None):
        super().__init__(agent)
        self.state = None
        self.last_stoh_k = None

    def tick_price(self, data):
        pass

    def kline_price(self, data):
        t = int(data['t'])
        o = float(data['o'])
        h = float(data['h'])
        l = float(data['l'])
        c = float(data['c'])
        self.bartimelist.append(t)
        self.bardict[t] = {'o': o, 'h': h, 'l': l, 'c': c, 'stoh_k':0}
        num1mforstoh = self.parameters['period'] * self.parameters['stoh_k']

        if len(self.bartimelist) > num1mforstoh:
            max_k = np.amax([self.bardict[t]['h'] for t in self.bartimelist[-num1mforstoh-1:-1]])
            min_k = np.amin([self.bardict[t]['l'] for t in self.bartimelist[-num1mforstoh-1:-1]])
            last_c = self.bardict[self.bartimelist[-2]]['c']
            stoh_k = (last_c - min_k) * 100 / (max_k - min_k)
            self.bardict[t]['stoh_k'] = stoh_k

            if self.last_stoh_k:
                #   если нету открытых ордеров
                if not self.currentcontract:
                    #   если стохастик пересек снизу уровень1 открываем BUY
                    if stoh_k > self.parameters['level1'] and self.last_stoh_k <= self.parameters['level1']:
                        # print(t, px, 'BUY level 1', self.last_stoh_k, stoh_k)
                        self.placeOrder(t, 'BUY', o, self.parameters['qty'])
                    #   если стохастик пересек сверху 100 - уровень1 открываем SELL
                    elif stoh_k < 100 - self.parameters['level1'] and self.last_stoh_k >= 100 - self.parameters['level1']:
                        #   открываем BUY
                        # print(t, px, 'SELL level 1', self.last_stoh_k, stoh_k)
                        self.placeOrder(t, 'SELL', o, self.parameters['qty'])
                #   если есть ордер на покупку BUY
                elif self.currentcontract.side == 'BUY':
                    #   если стохастик пересек сверху уровень1 закрываем всё
                    if stoh_k < self.parameters['level1'] and self.last_stoh_k >= self.parameters['level1']:
                        # print(t, px, 'CLOSE BUY level 1', self.last_stoh_k, stoh_k)
                        self.closeContract(t, o, self.currentcontract.qty)
                    #   если стохастик пересек снизу уровень2 закрываем половину
                    elif stoh_k > self.parameters['level2'] and self.last_stoh_k <= self.parameters['level2']:
                        pass
                        #   доп условие, если ордер пересекает уровень 2 в первый раз
                        # if self.currentcontract.chainnumber == 0:
                        #     # print(t, px, 'CLOSE BUY level 2', self.last_stoh_k, stoh_k)
                        #     self.closeContract(t, o, self.currentcontract.qty / 2)
                    #   если стохастик пересек снизу уровень3 закрываем dct
                    elif stoh_k > self.parameters['level3'] and self.last_stoh_k <= self.parameters['level3']:
                        # print(t, px, 'CLOSE BUY level 3', self.last_stoh_k, stoh_k)
                        self.closeContract(t, o, self.currentcontract.qty)
                #   если есть ордер на покупку BUY
                elif self.currentcontract.side == 'SELL':
                    # если стохастик пересек снизу 100 - уровень1 закрываем всё
                    if stoh_k > 100 - self.parameters['level1'] and self.last_stoh_k <= 100 - self.parameters['level1']:
                        # print(t, px, 'CLOSE SELL level 1', self.last_stoh_k, stoh_k)
                        self.closeContract(t, o, self.currentcontract.qty)
                    #   если стохастик пересек сверху 100 - уровень2 закрываем половину
                    elif stoh_k < 100 - self.parameters['level2'] and self.last_stoh_k >= 100 - self.parameters['level2']:
                        pass
                        #   доп условие, если ордер пересекает уровень 2 в первый раз
                        # if self.currentcontract.chainnumber == 0:
                        #     # print(t, px, 'CLOSE SELL level 2', self.last_stoh_k, stoh_k)
                        #     self.closeContract(t, o, self.currentcontract.qty / 2)
                    #   если стохастик пересек сверху 100 - уровень3 закрываем все
                    elif stoh_k < 100 - self.parameters['level3'] and self.last_stoh_k >= 100 - self.parameters['level3']:
                        # print(t, px, 'CLOSE SELL level 3', self.last_stoh_k, stoh_k)
                        self.closeContract(t, o, self.currentcontract.qty)

            self.last_stoh_k = stoh_k

    def tick_orderbook(self, data):
        pass

    def placeOrder(self, date, side, px, qty):
        self.lock.acquire()
        self.currentcontract = Contract(opentime=date, side=side, px=px, qty=qty)
        self.lock.release()
        # self.agent.placeOrder(side=side, px=px, qty=qty)

    def cancelOrder(self, orderid):
        pass

    def closeContract(self, date, px, qty):
        self.lock.acquire()
        if self.currentcontract.side == 'BUY':
            # profit = (px - self.currentcontract.px) * self.currentcontract.qty * self.parameters['koef']
            profit = (px - self.currentcontract.openpx) * qty * self.parameters['koef']
        else:
            # profit = (self.currentcontract.px - px) * self.currentcontract.qty * self.parameters['koef']
            profit = (self.currentcontract.openpx - px) * qty * self.parameters['koef']
        closedcontract = ClosedContract(opentime=self.currentcontract.opentime,
                                             side=self.currentcontract.side,
                                             openpx=self.currentcontract.openpx,
                                             qty=self.currentcontract.qty,
                                             closetime=date,
                                             closepx=px,
                                             profit=profit)

        self.closedcontractlist[date] = closedcontract
        self.parameters['balance'] += profit
        self.balancehistory[date] = self.parameters['balance']
        if qty < self.currentcontract.qty:
            # self.currentcontract.opentime = date
            # self.currentcontract.px = px
            self.currentcontract.qty -= qty
            self.currentcontract.chainnumber += 1
        else:
            self.currentcontract = None
        self.lock.release()
        # self.agent.closeContract(profit, self.parameters['balance'])