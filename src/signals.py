import numpy as np

from utils import *

def compute_sto_oscillator(close, high, low, lookback=14):
    '''Stochastic oscillator (SO)

    SO measures the momentum of security price to determine trends and indicate price reversal

    Parameters
    ----------
    close: pd.DataFrame
    high: pd.DataFrame
    lookback: int

    Return
    ------
    so: np.array
    '''
    recent_lowest = low.rolling(window=lookback, center=False).min()
    recent_highest = high.rolling(window=lookback, center=False).max()

    so = ((close - recent_lowest) / (recent_highest - recent_lowest))

    return so.values


def compute_bb(close, lookback=5, rank=False, norm=True):
    '''Bollinger Band (BB)
    
    BB generates overbought/oversold signals when prices toucher the upper/lower band

    Parameters
    ----------
    close: pd.DataFrame
    lookback: int

    Return
    ------
    bband: np.array
    '''
    def compute_sma(close, lookback):
        sma = close.rolling(window=lookback, center=False).mean()
        ratio = close/sma
        return sma, ratio
    
    sma, _ = compute_sma(close, lookback)
    rolling_std = close.rolling(window=lookback, center=False).std()
    
    bband = ((close - sma) /(2*rolling_std)).values
    bband = np.nan_to_num(bband, posinf=0, neginf=0)

    if rank:
        np.apply_along_axis(rank_stocks, arr=bband, axis=1)

    if norm:
        bband = normalize_signal(bband)

    return bband


def compute_neg_ret(price, rank=False, scale_factor=None, norm=True):
    '''Momentum

    Negative return as a short-term reversal signal 

    Parameters
    ----------
    price: pd.DataFrame
    rank: bool
    scale_factor: np.array

    Return
    ------
    neg_ret: np.array
    '''
    neg_ret = -1 * price.pct_change().values

    if rank:
        np.apply_along_axis(rank_stocks, arr=neg_ret, axis=1)
    
    if scale_factor is not None:
        neg_ret = scale_signal(neg_ret, scale_factor)
    
    if norm:
        neg_ret = normalize_signal(neg_ret)
    
    return neg_ret


def compute_macd(price, fast_period=12, slow_period=26, norm_period=20, 
                 signal_period=9, smooth_period=5, smooth=False, shift=0, norm=True):
    '''Moving Average Convergence Divergence (MACD)
    
    MACD is trend-following, by measuring the relationship between 2 MAs of the security price

    Return
    ------
    signal: np.array
    '''
    ema_fast = price.ewm(span=fast_period).mean().shift(shift)
    ema_slow = price.ewm(span=slow_period).mean().shift(shift)
    
    macd = ema_fast - ema_slow
    recent_max_macd = macd.rolling(window=norm_period, center=False).max()
    recent_min_macd = macd.rolling(window=norm_period, center=False).min()
    
    macd_normalized = (macd - recent_min_macd) / (recent_max_macd - recent_min_macd)
    macd_final = macd_normalized.ewm(span=smooth_period).mean() if smooth else macd_normalized
    signal = macd_final.ewm(span=signal_period).mean().values
    
    if norm: 
        signal = normalize_signal(signal)

    return signal


# Combine signals
def add(indicators, norm=False):
    '''Add all signals
    '''
    ind = sum(indicators)
    if norm:
        ind = normalize_signal(ind)
    return ind


def avg(indicators, norm=False):
    '''Average across all signals
    '''
    ind = np.mean(indicators, axis=0)
    if norm:
        ind = normalize_signal(ind)
    return ind
