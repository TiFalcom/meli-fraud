import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
import lightgbm as lgbm
import optuna
from src.utils.modeling import objective
import pickle

@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/encoded.')
@click.option('--hparams_file_prefix', default='hp', type=str, help='Hiperparams file prefix, to save output params.')
def main(dataset_prefix, hparams_file_prefix):

    logger = logging.getLogger('Tune-Optuna')

    logger.info(f'Loading {dataset_prefix} train dataset.')
    df_train = pd.read_parquet(os.path.join('data', 'encoded', f'{dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} train: {df_train.shape}')

    logger.info(f'Loading {dataset_prefix} valid dataset.')
    df_valid = pd.read_parquet(os.path.join('data', 'encoded', f'{dataset_prefix}_valid.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} valid: {df_valid.shape}')

    selector = pickle.load(open('model/encoders/selector.pkl', 'rb'))

    sampler = optuna.samplers.TPESampler(seed=777)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    logger.info('Start tunning.')
    # Don't use n_jobs to keep replicability
    study.optimize(lambda trial: objective(trial,
                                        selector.transform(df_train), 
                                        df_train['fraude'],
                                        [(selector.transform(df_valid), 
                                            df_valid['fraude'])]), n_trials=15, n_jobs=1)

    logger.info(f'Best parms: {study.best_params}')

    yaml.safe_dump(study.best_params, open(f'src/model/config/{hparams_file_prefix}_hparams.yml', 'w'))

    logger.info(f'Success!')
    

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()