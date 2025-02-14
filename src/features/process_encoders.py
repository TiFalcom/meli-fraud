import pandas as pd
import numpy as np
import logging
import click
import os
import pickle


@click.command()
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/enrich.')
@click.option('--dataset_prefix_output', default=None, type=str, help='Data set name on data/enrich.')
def main(dataset_prefix, dataset_prefix_output):

    logger = logging.getLogger('Process-Encoders')

    ord_enc = pickle.load(open('model/encoders/ordinal_encoder.pkl', 'rb'))

    if not dataset_prefix_output:
        dataset_prefix_output = dataset_prefix

    for group in ['train', 'test', 'valid']:
        logger.info(f'Loading {dataset_prefix} {group} dataset.')
        df = pd.read_parquet(os.path.join('data', 'enrich', f'{dataset_prefix}_{group}.parquet.gzip'))
        logger.info(f'Shape {dataset_prefix} {group}: {df.shape}')

        logger.info(f'Starting Encoding')
        df = ord_enc.transform(df)
        logger.info(f'Encoding Completed! Shape: {df.shape}')

        logger.info('Saving dataset on data/encoded')
        df.to_parquet(
            os.path.join('data', 'encoded', f'{dataset_prefix_output}_{group}.parquet.gzip'),
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