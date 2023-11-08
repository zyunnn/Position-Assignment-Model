import numpy as np
import argparse

def check_missing_vals(dfs):
    counter = {field: df.isnull().sum(axis=1).sum() for field, df in dfs.items()}
    return counter


def scale_signal(signals, scale_factor):
    '''Scale signal by factor

    Parameters
    ----------
    signals: np.array
    scale_factor: np.array
    '''
    
    def scale(arr, n):
        signal = arr[:n]
        vol = arr[n:]
        
        long = signal > 0
        short = signal < 0
        
        long_vol = vol[long]
        short_vol = vol[short]

        signal[long] *= long_vol / sum(long_vol)
        signal[short] *= short_vol / sum(short_vol)
        
        return signal
    
    arr = np.hstack((signals, scale_factor))
    return np.apply_along_axis(scale, 1, arr, signals.shape[1])


def normalize_signal(signals):
    '''Normalize signal within long/short bucket

    1. Rank stocks by signal score to assign names into long/short bucket
    2. Normalized signal score by sum of signal score in each bucket
    
    This ensures that overall net exposure is (close to) zero

    Parameters
    ----------
    signals: np.array
    '''
    def norm(arr, n):
        signal = arr[:n]
        ranks = arr[n:]
        
        long = ranks > 0
        short = ranks < 0
        
        signal[long] = np.abs(signal[long]) / sum(np.abs(signal[long]))
        signal[short] = -np.abs(signal[short]) / sum(np.abs(signal[short]))
        
        return signal

    ranks = signals.copy()
    np.apply_along_axis(rank_stocks, arr=ranks, axis=1)

    arr = np.hstack((signals, ranks))
    return np.apply_along_axis(norm, 1, arr, signals.shape[1])


# TODO: split into n buckets
def rank_stocks(row):
    '''Rank stock by signal score

    By default, split all stocks into 2 buckets and long(short) 
    the bottom(top) bucket for reversal signals
    '''
    if np.isnan(row).all():
        return 
    
    n = len(row)
    sorted_idx = np.argsort(row)[::-1]
    top = sorted_idx[:n//2]
    bottom = sorted_idx[n//2:]
    row[top] = -1
    row[bottom] = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Specify filename for results', required=True)
    parser.add_argument('--output_dir', default='results', help='Specify directory to save results')
    return parser.parse_args()


