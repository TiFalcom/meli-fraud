import pandas as pd
import numpy as np
import logging
import click
import os
import yaml
from sklearn.ensemble import RandomForestClassifier
from feature_engine.encoding import OrdinalEncoder


@click.command()
@click.option('--config_file', default='features', type=str, help='Features configuration file on src/data/config.')
@click.option('--dataset_prefix', default=None, type=str, help='Data set name on data/processed.')
@click.option('--fs_file_prefix', default='rf', type=str, help='Feature Selection file prefix, to save output features.')
def main(config_file, dataset_prefix, fs_file_prefix):

    logger = logging.getLogger('Feature-Store')

    hard_remove_features = yaml.safe_load(open(f'src/data/config/{config_file}.yml', 'r'))['hard_remove_features']

    logger.info(f'Loading {dataset_prefix} train dataset.')
    df = pd.read_parquet(os.path.join('data', 'enrich', f'{dataset_prefix}_train.parquet.gzip'))
    logger.info(f'Shape {dataset_prefix} train: {df.shape}')

    logger.info(f'Generating random features')
    for i in range(1,3):
        # replicability
        np.random.seed(i)
        df[f'___random_cat_{i}___'] = np.random.randint(0, i*25, size=df.shape[0])

        np.random.seed(i)
        df[f'___random_con_{i}___'] = np.random.uniform(0, 1000000, size=df.shape[0])

    logger.info(f'Fitting Encoder')
    ord_enc = OrdinalEncoder(
        encoding_method='arbitrary',
        variables=df.drop(columns=hard_remove_features).select_dtypes(include=['object']).columns.tolist()
    ).fit(df)

    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5,
                            max_features='sqrt', class_weight='balanced', random_state=777, n_jobs=-1)
    
    logger.info(f'Training Random Forest')
    rf.fit(ord_enc.transform(df).drop(columns=hard_remove_features), df['fraude'])
    logger.info(f'Random Forest Finished!')

    df_importance = pd.DataFrame({
        'feature': rf.feature_names_in_,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    df_importance['importance_cumsum'] = df_importance['importance'].cumsum()
    logger.info(f'Feature Importance:')
    print(df_importance)

    features_support = df_importance[df_importance['importance_cumsum'] < 0.96]['feature'].tolist()
    features_reject = list(set(df_importance['feature'].tolist()) - set(features_support))

    logger.info(f'Features selected: {features_support}')

    features_selected_yml = {
        'support' : features_support,
        'reject' : features_reject,
        'hard_remove' : hard_remove_features
    }

    logger.info(f'Saving artifacts')
    yaml.safe_dump(features_selected_yml, open(f'src/features/config/{fs_file_prefix}_selection.yml', 'w'))

    df_importance.to_csv(f'reports/{fs_file_prefix}_importance._csv', index=False)

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