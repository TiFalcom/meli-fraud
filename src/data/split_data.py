import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_name', default=None, type=str, help='Data set name on data/interim.')
@click.option('--ymd_train', default=None, type=str, help='YYYY-MM-DD from beggining of the training dataset, included')
@click.option('--ymd_test', default=None, type=str, help='YYYY-MM-DD from beggining of the testing dataset, included')
def main(config_file, dataset_name, ymd_train, ymd_test):

    logger = logging.getLogger('Split-Data')
    
    ymd_train = datetime.strptime(ymd_train, '%Y%m%d')
    ymd_test = datetime.strptime(ymd_test, '%Y%m%d')

    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))
    temporal_feature = config_features['temporal_feature']
    target_column = config_features['target_feature']

    logger.info(f'Loading {dataset_name} dataset.')
    df = pd.read_csv(os.path.join('data', 'raw', f'{dataset_name}.csv'), index_col=0).reset_index(drop=True)
    logger.info(f'Shape {dataset_name}: {df.shape}')

    df_train = df[
        (df[temporal_feature].astype('datetime64[s]') >= ymd_train)
        & (df[temporal_feature].astype('datetime64[s]') < ymd_test)
    ].reset_index(drop=True)

    index_train, index_valid = train_test_split(df_train.index, test_size=0.2, stratify=df_train[target_column])

    index_test = df[
        (df[temporal_feature].astype('datetime64[s]') >= ymd_test)
    ].index

    logger.info(f'Saving train dataset. Shape: {df_train.iloc[index_train].shape}')
    df_train.iloc[index_train].to_parquet(
        os.path.join('data', 'raw', f'{dataset_name}_train.parquet.gzip'),
        compression='gzip',
        index=False
    )
    logger.info('Success!')

    logger.info(f'Saving test dataset. Shape: {df.iloc[index_test].shape}')
    df.iloc[index_test].to_parquet(
        os.path.join('data', 'raw', f'{dataset_name}_test.parquet.gzip'),
        compression='gzip',
        index=False
    )
    logger.info('Success!')

    logger.info(f'Saving valid dataset. Shape: {df_train.iloc[index_valid].shape}')
    df_train.iloc[index_valid].to_parquet(
        os.path.join('data', 'raw', f'{dataset_name}_valid.parquet.gzip'),
        compression='gzip',
        index=False
    )
    logger.info('Success!')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()