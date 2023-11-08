from abc import ABC, abstractmethod
import numpy as np

from models.backtest import PnLCalculator

class Allocator(ABC):
    '''
    Abstract class for all position assignment model
    '''

    def __init__(self, data):
        self.data = data
        self.signals = []
        self.final_signal = None
        self.allocations = None
        self.pnl_cal = None
        
    @abstractmethod
    def run(self, *args):
        pass

    def add_method(self, price, func, args):
        '''Create signal

        Parameters
        ----------
        price: pd.DataFrame
        func: function
        args: dict
        '''
        indicator = func(price, **args)
        if self.final_signal is None:
            self.final_signal = indicator
        self.signals.append(indicator)


    def combine(self, func):
        '''Combine signals using user-defined function
        '''
        self.final_signal = func(self.signals)


    def allocate(self, notional):
        '''Create signal and scale by notional value

        Parameters
        ----------
        notional: np.float
        
        Return
        ------
        allocation: np.array
        '''
        self.allocations =  self.final_signal * notional

        # Liquidate all positions, avoid overnight risk
        dates = self.data['open'].index.floor('D')
        mask = ~dates.duplicated(keep='last')
        self.allocations[mask, ] = np.nan

        self.check_net_exposure(self.allocations)
        
        return self.allocations 
    

    def check_net_exposure(self, allocation, thres=1e-6):
        '''Ensure zero net exposure

        Parameters
        ----------
        allocation: np.array
        thres: np.float
        '''
        assert(np.nanmax(allocation.sum(axis=1)) < thres)


    def evaluate(self):
        '''Evaluate performance using realized price data
        '''
        self.pnl_cal = PnLCalculator(self.data['open'], self.data['close'])
        self.performance = self.pnl_cal.trade_frictionless(self.allocations)
        self.pnl_cal.plot_results()

        # Number of open position
        pos = np.abs(self.allocations)
        daily_pos = ((pos != 0) & (~np.isnan(pos))).sum(axis=1)

        # Print portfolio summary
        summary = self.pnl_cal.summary()
        summary['Avg # position'] = daily_pos.mean()
        print(summary)

        return self.performance, summary


    