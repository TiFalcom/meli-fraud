import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
from feature_engine.encoding import OrdinalEncoder
from src.utils.transformers import Selector, FeatureEngineering, FixFeaturesMissing, FixFeaturesType
import pickle


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/processed.')
@click.option('--fs_file_prefix', default='rf', type=str, help='Feature Selection file prefix, to save output features.')
def main(config_file, dataset_prefix, fs_file_prefix):

    logger = logging.getLogger('Encoding')

    config_features = yaml.safe_load(open(f'src/data/config/{config_file}.yml', 'r'))
    support_features = yaml.safe_load(open(f'src/features/config/{fs_file_prefix}_selection.yml', 'r'))['support']

    logger.info(f'Loading {dataset_prefix} train dataset.')
    df = pd.read_parquet(os.path.join('data', 'enrich', f'{dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} train: {df.shape}')

    logger.info(f'Fitting Encoders')
    ord_enc = OrdinalEncoder(
        encoding_method='arbitrary',
        variables=df[support_features].select_dtypes(include=['object', 'string']).columns.tolist()
    ).fit(df)

    selector = Selector(support_features).fit(df)
    fix_type = FixFeaturesType(config_features['fix_type_map']).fit(df)
    fix_missing = FixFeaturesMissing(config_features['fix_missing_numeric_features'], 
                                    config_features['fix_missing_map']).fit(df)
    feature_engineering = FeatureEngineering().fit(df)

    logger.info(f'Saving Encoders on model/encoders')
    pickle.dump(ord_enc, open('model/encoders/ordinal_encoder.pkl', 'wb'))
    pickle.dump(selector, open('model/encoders/selector.pkl', 'wb'))
    pickle.dump(fix_type, open('model/encoders/fix_type.pkl', 'wb'))
    pickle.dump(fix_missing, open('model/encoders/fix_missing.pkl', 'wb'))
    pickle.dump(feature_engineering, open('model/encoders/feature_engineering.pkl', 'wb'))

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