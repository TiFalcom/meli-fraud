import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
import pickle

@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/encoded.')
@click.option('--hparams_file_prefix', default='hp', type=str, help='Hiperparams file prefix, to save output params.')
@click.option('--model_prefix', default='lgbm', type=str, help='Model name prefix, to be saved.')
def main(dataset_prefix, hparams_file_prefix, model_prefix):

    logger = logging.getLogger('Tune-Optuna')

    logger.info(f'Loading {dataset_prefix} train dataset.')
    df_train = pd.read_parquet(os.path.join('data', 'encoded', f'{dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} train: {df_train.shape}')

    logger.info(f'Loading {dataset_prefix} valid dataset.')
    df_valid = pd.read_parquet(os.path.join('data', 'encoded', f'{dataset_prefix}_valid.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} valid: {df_valid.shape}')

    selector = pickle.load(open('model/encoders/selector.pkl', 'rb'))
    
    params = yaml.safe_load(open(f'src/model/config/{hparams_file_prefix}_hparams.yml', 'r'))

    logger.info('Start Training.')
    model = lgbm.LGBMClassifier(
        verbose = 0,
        random_state = 777,
        importance_type = 'gain',
        # Early stopping changed to 2, to reduce overfitting
        early_stopping_rounds = 2,
        objective= 'binary',
        boosting_type= 'gbdt',
        n_jobs=-1,
        **params
    )

    model.fit(selector.transform(df_train),
            df_train['fraude'],
            eval_set=[(
            selector.transform(df_valid),
            df_valid['fraude']
            )], 
            eval_metric='auc'
    )

    logger.info('Training finished!')

    y_pred_train = model.predict_proba(selector.transform(df_train))[:,1]
    y_pred_valid = model.predict_proba(selector.transform(df_valid))[:,1]

    logger.info(f"ROCAUC Train: {roc_auc_score(df_train['fraude'], y_pred_train)}")
    logger.info(f"ROCAUC Valid: {roc_auc_score(df_valid['fraude'], y_pred_valid)}\n")

    logger.info(f"ROCAUC@1% Train: {roc_auc_score(df_train['fraude'], y_pred_train, max_fpr=0.01)}")
    logger.info(f"ROCAUC@1% Valid: {roc_auc_score(df_valid['fraude'], y_pred_valid, max_fpr=0.01)}")

    logger.info('Saving model.')
    pickle.dump(model, open(f'model/predictors/{model_prefix}_model.pkl', 'wb'))
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