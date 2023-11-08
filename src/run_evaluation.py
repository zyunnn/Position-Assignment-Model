import os
import pandas as pd

from data.dataloader import DataLoader
from models.backtest import PnLCalculator
from utils import get_args


if __name__ == '__main__':

    args = get_args()
    
    # load raw data
    loader = DataLoader()
    intraday_data = loader.load_raw_data(fillna=True)

    # load assignment
    fname = os.path.join('results', args.file)
    pos_df = pd.read_csv(fname, index_col=0)
    calc = PnLCalculator(intraday_data['open'], intraday_data['close'])
    performance = calc.trade_frictionless(pos_df.values)
    print(calc.summary())

    # save evaluation result
    calc.plot_results(save_png=True, dir=args.output_dir)
    print(f'Save to ../{args.output_dir}')
