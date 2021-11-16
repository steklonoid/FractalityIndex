from threading import Thread
import numpy as np


class LocalFile(Thread):
    #   читает файлы, полученные с https://data.binance.vision/
    #   формат файлов: CSV - файлы, значения разделенные запятыми,
    #   значения полей слева направо:
    #   Open time /	Open / High / Low / Close / Volume / Close time / Quote asset volume / Number of trades / Taker buy base asset volume / Taker buy quote asset volume / Ignore
    def __init__(self, q):
        super(LocalFile, self).__init__()
        self.q = q
        self.parameters = {'filename':'', 'timeinterval':'', 'coin': ''}

    def run(self) -> None:
        ar = np.genfromtxt(self.parameters['filename'], delimiter=',')
        ar[:, 0] //= 1000
        for i in range(ar.shape[0]):
            self.q.put({'t':ar[i, 0] , 'o':ar[i, 1], 'h':ar[i, 2], 'l':ar[i, 3], 'c':ar[i, 4]})
        self.q.put(None)

    def placeOrder(self, side, px, qty):
        pass

    def closeContract(self, profit, balance):
        pass