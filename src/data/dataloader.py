import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import zipfile


year = '2023'

class DataLoader:

    def __init__(self, file='data/sample_data.zip'):
        self.file = file

    @staticmethod
    def _forward_fill(dfs):
        filled_dfs = {field: df.ffill() for field, df in dfs.items()}
        return filled_dfs   

    def load_raw_data(self, fillna=True):
        '''Load raw data from zip file

        Returns
        -------
        dfs: dict
            Dictionary of dataframes with raw market data
        '''
        daily_data = defaultdict(list)
        with zipfile.ZipFile(self.file) as f:
            files = f.namelist()
            for fname in files:
                field, dt = fname.split('.')[0].rsplit('_', 1)
                df = pd.read_csv(f.open(fname))
                daily_data[field].append((dt, df))
        print('Load data successfully!')

        # format dataframes
        dfs = dict.fromkeys(daily_data.keys(), pd.DataFrame())
        for field, df_ls in tqdm(daily_data.items()):
            for dt, df in df_ls:
                df['Date'] = year + dt + ' ' + df['Minutes']
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df.drop(columns='Minutes', axis=1)
                dfs[field] = pd.concat([dfs[field], df])
            dfs[field].sort_index(inplace=True)

        if fillna:
            dfs = self._forward_fill(dfs)

        return dfs
    
    

    

