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

