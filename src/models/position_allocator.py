from models.base import Allocator
from signals import *


class UnivariateModel(Allocator):
    '''Best performing univariate model
    
    Bollinger band as short-term reversal signal
    '''

    def __init__(self, intraday_data, notional=1_000_000):
        super().__init__(intraday_data)
        self.notional = notional

    def run(self):
        self.add_method(self.data['vwap'], compute_bb, {'lookback': 5, 'rank': True, 'norm': True})
        _ = self.allocate(self.notional)



class MultivariateModel(Allocator):
    '''Best performing multivariate model
    
    Combine Bollinger band and Momentum
    '''
    
    def __init__(self, intraday_data, notional=1_000_000):
        super().__init__(intraday_data)
        self.notional = notional

    def run(self):
        self.add_method(self.data['vwap'], compute_bb, {'lookback': 5, 'rank': True, 'norm': True})
        self.add_method(self.data['close'], compute_neg_ret, {'rank': False})
        self.combine(avg)
        _ = self.allocate(self.notional)


class TestModel(Allocator):
    '''Toy model to test different signal-weighting scheme
    '''

    def __init__(self, intraday_data, notional=1_000_000):
        super().__init__(intraday_data)
        self.notional = notional

    def run(self, methods=[], c_func=None):
        for col, func, args in methods:
            self.add_method(self.data[col], func, args)

        if c_func is not None:
            self.combine(c_func)

        _ = self.allocate(self.notional)
