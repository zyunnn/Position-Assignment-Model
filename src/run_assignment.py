import os
import pandas as pd

from data.dataloader import DataLoader
from models.position_allocator import MultivariateModel
from utils import get_args


if __name__ == '__main__':

    args = get_args()
    
    # load raw data
    loader = DataLoader()
    intraday_data = loader.load_raw_data(fillna=True)

    # run model to get allocation
    model = MultivariateModel(intraday_data)
    model.run()
    target_position = model.allocations

    # save result to csv file
    fname = os.path.join(args.output_dir, args.file)
    df = intraday_data['open']
    pd.DataFrame(target_position, columns=df.columns, index=df.index).to_csv(fname)
    print(f'Save results to {fname}')

