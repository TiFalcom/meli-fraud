import pandas as pd
import numpy as np
from datetime import datetime

class FeatureStore:

    key = 'j'
    date_ref = 'ymd'
    window = 7
    delay = 1
    name = None
    path_to_save = None

    def __repr__(self):
        return f'I am {self.name}'

    @classmethod
    def save(self):
        self.data.to_parquet(
            f'{self.path_to_save}/{self.name}.parquet.gzip',
            compression='gzip',
            index=False
        )
        return self

    @classmethod
    def load(self):
        self.data = pd.read_parquet(f'{self.path_to_save}/{self.name}.parquet.gzip')
        return self

    @classmethod
    def process(self, X):
        print('Hi! Implement this method!')
        return self

    @classmethod
    def merge(self, X):
        return X.merge(
            self.data,
            how='left',
            on=[self.key, self.date_ref]
        )


class FeatureStoreCategoryProfile(FeatureStore):

    name = 'FeatureStoreCategoryProfile'
    path_to_save = 'data/feature_store'

    @classmethod
    def process(self, df):
        # Fill dataset with all possible dates
        dt_init = str(df['ymd'].min())
        dt_end = str(df['ymd'].max())
        all_dates = pd.date_range(start=datetime.strptime(dt_init, '%Y%m%d'), end=datetime.strptime(dt_end, '%Y%m%d'))

        unique_clients = df['j'].unique()
        df_clients_with_all_dates = pd.MultiIndex.from_product([unique_clients, all_dates], names=['j', 'ymd']).to_frame(index=False)
        df_clients_with_all_dates['ymd'] = df_clients_with_all_dates['ymd'].dt.strftime('%Y%m%d').astype(int)

        df_processed = df_clients_with_all_dates.merge(df[['j', 'ymd', 'monto', 'fraude']], on=['j', 'ymd'], how='left')
        
        # Create new Features
        df_processed['monto_fraude'] = df_processed['monto'] * df_processed['fraude']

        # Aggregate
        self.data = (
            df_processed
            .groupby(['j', 'ymd'])
            .agg({
                'monto' : ['sum', 'count'],
                'monto_fraude' : 'sum',
                'fraude' : 'sum'
            })
            .sort_values(by=['j', 'ymd'], ascending=True)
            .reset_index()
            .groupby('j')
            .rolling(window=self.window, min_periods=1)
            .agg({
                ('ymd', '') : 'max',
                ('monto', 'sum') : 'sum',
                ('monto', 'count') : 'sum',
                ('monto_fraude', 'sum') : 'sum',
                ('fraude', 'sum') : 'sum'
            })
            .reset_index(level=0)
        )
        # Fix date ref, to not include current day
        self.data['ymd'] += self.delay
        self.data.columns = ['j', 'ymd', 'vl_cat_last7d', 'qty_cat_last7d', 'vl_fraud_cat_last7d', 'qty_fraud_cat_last7d']

        # New features to this feature store
        self.data['br_vl_cat_last7d'] = np.nan_to_num(self.data['vl_fraud_cat_last7d'] / self.data['vl_cat_last7d'], 
                                                       nan=0, posinf=1e9, neginf=-1e9)
        self.data['br_qty_cat_last7d'] = np.nan_to_num(self.data['qty_fraud_cat_last7d'] / self.data['qty_cat_last7d'], 
                                                       nan=0, posinf=1e9, neginf=-1e9)
        
        return self