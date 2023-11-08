import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PnLCalculator:
    '''
    Evaluate performance of position assignment model

    Key assumptions:
    - On each trading day, clear all EOD position by 16:00
    - Rebalance every minute to ensure zero net exposure
    - Adjust position at next `open` price
    '''

    def __init__(self, open, close):
        self.open = open
        self.close = close
        self.performance = pd.DataFrame(index=np.unique(self.open.index.date), 
                                        columns=['Daily PnL', 'Wealth', 'Long', 'Short', 
                                                 'Max Drawdown', 'Daily vol', 'Daily turnover'])


    def trade_frictionless(self, allocations):
        '''Evaluate performance in frictionless market

        Parameters
        ----------
        alloc: np.array

        Return
        ------
        performance: pd.DataFrame
            List of evaluation metrics 
        '''
        profit = 0
        
        allocations = pd.DataFrame(allocations, index=self.open.index, columns=self.open.columns)

        # enter and exit price for current position
        open_df = self.open.shift(-1)
        close_df = self.close.shift(-1)

        for (dt, alloc), (_, close), (_, open) in zip(allocations.groupby(allocations.index.date), 
                                                       close_df.groupby(close_df.index.date), 
                                                       open_df.groupby(open_df.index.date)):
            # compute pnl
            pos_chg = (alloc/open).fillna(0).diff()
            price_chg = close - open
            pnl = pos_chg * price_chg
            profit += pnl.sum(axis=1).sum()

            # evaluation metrics
            self.performance.loc[dt, 'Daily PnL'] = pnl.sum(axis=1).sum()
            self.performance.loc[dt, 'Wealth'] = profit
            self.performance.loc[dt, 'Max Drawdown'] = self._max_drawdown(pnl)
            self.performance.loc[dt, 'Daily turnover'] = self._turnover(alloc)

            exp_rtn, std_rtn = self._return_vol(alloc.values, pnl.values)
            self.performance.loc[dt, 'Daily return'] = exp_rtn
            self.performance.loc[dt, 'Daily vol'] = std_rtn

            long_profit, short_profit  = self._bucket_pnl(alloc.values, pnl.values)
            self.performance.loc[dt, 'Long'] = long_profit
            self.performance.loc[dt, 'Short'] = short_profit
            
        return self.performance
    
    def summary(self):
        '''Portfolio summary 
        '''
        portfolio_summary = {
            'Sharpe': (self.performance['Daily return'] / self.performance['Daily vol']).mean() * np.sqrt(252),
            # 'Ann. vol': self.performance['Daily vol'].mean() * np.sqrt(252),
            'MDD': self.performance['Max Drawdown'].min(),
            'Max turnover': self.performance['Daily turnover'].max()
        }
        return portfolio_summary


    def plot_results(self, save_png=False, dir='results'):
        '''Visualize portfolio performance

        Parameters
        ----------
        save_png: bool
        dir: str
        '''
        cols = ['Daily PnL', 'Daily return', 'Daily vol', 'Daily turnover', 'Wealth', 'Max Drawdown']
        # n = len(cols)+1
        fig, ax = plt.subplots(2, 4, figsize=(5*4, 4*2), sharex=True)
        ax = ax.flatten()

        self.performance['Long'].plot(ax=ax[0], alpha=0.5, label='Long', color='green')
        self.performance['Short'].plot(ax=ax[0], alpha=0.5, label='Short', color='orange')
        ax[0].set_title('Long/Short bucket')
        ax[0].legend()
        ax[0].tick_params(axis='x', labelrotation=45)

        for i, col in enumerate(cols):
            self.performance[col].plot(ax=ax[i+1])
            ax[i+1].set_title(col)
            ax[i+1].tick_params(axis='x', labelrotation=45)

        plt.suptitle('Portfolio Performance', size=14, y=1.05)
        plt.tight_layout()

        if save_png:
            plt.savefig(dir + '/performance.png', bbox_inches='tight')
        else:
            plt.show()


    def _bucket_pnl(self, alloc_arr, pnl_arr):
        '''Compute daily PnL for long/short bucket separately
        '''
        def masked_sum(arr, mask):
            return np.nansum(np.ma.masked_array(arr, ~mask).sum(axis=1))

        long_pos = masked_sum(pnl_arr,(alloc_arr > 0) & (alloc_arr != 0))
        short_pos = masked_sum(pnl_arr,(alloc_arr < 0) & (alloc_arr != 0))

        return long_pos, short_pos
    

    def _return_vol(self, alloc_arr, pnl_arr):
        '''Compute daily portfolio return and volatility
        '''
        daily_rtn = np.nan_to_num(pnl_arr / np.abs(alloc_arr), neginf=0, posinf=0).sum(axis=1)
        return daily_rtn.mean(), daily_rtn.std()


    def _max_drawdown(self, daily_pnl, w=60):
        ''' Compute daily maximum drawdown
        '''
        peak = daily_pnl.sum(axis=1).rolling(window=w).max()
        trough = daily_pnl.sum(axis=1).rolling(window=w).min()       
        mdd = (trough - peak) / peak
        return mdd.max()
    

    def _turnover(self, alloc):
        '''Compute daily portfolio turnover
        '''
        turnover = np.abs(alloc.diff()).sum(axis=1).sum()
        return turnover
