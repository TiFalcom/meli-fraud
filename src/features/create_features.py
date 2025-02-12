import pandas as pd
import numpy as np
import logging
import click
import os
from src.utils.transformers import FeatureEngineering


@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/interim.')
def main(dataset_prefix):

    logger = logging.getLogger('Feature-Engineering')

    create_features = FeatureEngineering(inference=False)

    for group in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_prefix} {group} dataset.')
        df = pd.read_parquet(os.path.join('data', 'interim', f'{dataset_prefix}_{group}.parquet.gzip'))
        logger.info(f'Shape {dataset_prefix} {group}: {df.shape}')

        logger.info(f'Starting Feature Engineering')
        df = create_features.fit_transform(df)
        logger.info(f'Feature Enginerring Completed! Shape: {df.shape}')

        logger.info('Saving dataset on data/processed')
        df.to_parquet(
            os.path.join('data', 'processed', f'{dataset_prefix}_{group}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')

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