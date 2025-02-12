import pandas as pd
import logging
import click
import os
import yaml

from src.utils.transformers import FixFeaturesType, FixFeaturesMissing
from src.utils.modeling import apply_encoders


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_name', default=None, type=str, help='Data set name on data/raw.')
def main(config_file, dataset_name):

    logger = logging.getLogger('Basic-Process')
    
    config_features = yaml.safe_load(open(os.path.join('src', 'data', 'config', f'{config_file}.yml'), 'r'))

    logger.info(f'Loading {dataset_name} valid dataset.')
    df_valid = pd.read_parquet(os.path.join('data', 'raw', f'{dataset_name}_valid.parquet.gzip'))
    logger.info(f'Shape {dataset_name} valid: {df_valid.shape}')

    logger.info(f'Fit FixFeaturesType.')
    fix_type = FixFeaturesType(config_features['fix_type_map'])

    df_valid = fix_type.fit_transform(df_valid)

    logger.info(f'Fit FixFeaturesMissing.')

    fix_missing = FixFeaturesMissing(config_features['fix_missing_numeric_features'], 
                                     config_features['fix_missing_map']).fit(df_valid)
    
    logger.info(f'Transforming {dataset_name} datasets.')

    for group in ['train', 'valid', 'test']:

        logger.info(f'Loading {dataset_name} {group} dataset.')
        df = pd.read_parquet(os.path.join('data', 'raw', f'{dataset_name}_{group}.parquet.gzip'))
        logger.info(f'Shape {dataset_name} {group}: {df.shape}')

        df = apply_encoders(df, [fix_type, fix_missing])

        logger.info(f'Saving {dataset_name} {group} on data/interim')
        df.to_parquet(
            os.path.join('data', 'interim', f'{dataset_name}_{group}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Success!')

    logger.info('Success! All datasets processed!')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()