import pandas as pd
import numpy as np
import logging
import click
import os
from src.utils.feature_store import FeatureStoreCategoryProfile


@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/processed.')
def main(dataset_prefix):

    logger = logging.getLogger('Feature-Store')

    feature_stores = [
        FeatureStoreCategoryProfile
    ]

    for feature_store in feature_stores:
        logger.info(f'Loading {feature_store.name}')
        feature_store.load()
        logger.info(f'{feature_store.name} load!')

    for group in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_prefix} {group} dataset.')
        df = pd.read_parquet(os.path.join('data', 'processed', f'{dataset_prefix}_{group}.parquet.gzip'))
        logger.info(f'Shape {dataset_prefix} {group}: {df.shape}')

        for feature_store in feature_stores:
            logger.info(f'Merging {feature_store.name}')
            df = feature_store.merge(df)
            logger.info(f'{feature_store.name} merged!')

        logger.info(f'Saving dataset on data/enrich. Shape: {df.shape}')
        df.to_parquet(
            os.path.join('data', 'enrich', f'{dataset_prefix}_{group}.parquet.gzip'),
            compression='gzip',
            index=False
        )
        logger.info('Saved!')
    

    logger.info('Success! All Feature Stores Processed!')

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #filename='data/logs/basic_process.log',
        #filemode='w'
    )
    main()