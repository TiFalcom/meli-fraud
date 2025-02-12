import numpy as np
import pandas as pd
from datetime import datetime

class FixFeaturesType:
    """ Class to fix features's type
    """

    def __init__(self, cast_features):

        self.cast_features = cast_features

        self.type_function = {
            'float32' : np.float32,
            'float64' : np.float64,
            'str' : str,
            'int64' : np.int64,
            'int32' : np.int32,
            'datetime[64]' : 'datetime64[s]'
        }

    def fit(self, X, y=None):
        return self

    def fix_type(self, X):
        for type, columns in self.cast_features.items():
            # TODO: Change to loc
            X[columns] = X[columns].astype(self.type_function[type])
        return X

    def transform(self, X):
        X_tmp = X.reset_index(drop=True)
        return self.fix_type(X_tmp)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class FixFeaturesMissing:
    """ Class to fix features's missing values
    """
    def __init__(self, missing_numeric_features, missing_features_map):
        self.missing_numeric_features = missing_numeric_features
        self.missing_features_map = missing_features_map

    def __repr__(self):
        return f'Features to fix: {list(self.missing_features_map.keys())}'

    def fit(self, X, y=None):
        for feature in self.missing_numeric_features:
            self.missing_features_map[feature] = round(X[feature].median(), 4)
        return self
    
    def transform(self, X):
        for feature, missing_value in self.missing_features_map.items():
            X.loc[:, feature] = X.loc[:, feature].fillna(missing_value)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

def get_period_of_day(hour):
    if hour < 6:
        return 'dawn'
    if hour < 12:
        return 'morning'
    if hour < 18:
        return 'afternoon'
    if hour < 24:
        return 'evening'

class FeatureEngineering:
    """Class to create new features based on existing ones
    """
    def __init__(self, inference=True):
        self.inference = inference

    def fit(self, X, y=None):
        return self
    
    def create_train_features(self, X):
        X = X.reset_index(drop=True)
        X_tmp = X.reset_index(drop=True)

        X_tmp['ymd'] = X_tmp['fecha'].apply(lambda x: datetime.strftime(x, '%Y%m%d')).astype(int)

        #X_tmp['monto_fraude'] = X_tmp['fraude'] * X_tmp['monto_fraude']

        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], left_index=True, right_index=True, how='left')
        
        return X_tmp
    
    def create_payload_features(self, X):
        X = X.reset_index(drop=True)
        X_tmp = X.reset_index(drop=True)

        X_tmp['day_of_week'] = X_tmp['fecha'].dt.weekday

        X_tmp['hour_of_day'] = X_tmp['fecha'].dt.hour

        X_tmp['period_of_day'] = X_tmp['hour_of_day'].apply(get_period_of_day)

        new_columns = [col for col in X_tmp.columns if col not in X.columns]
        X_tmp = X.merge(X_tmp[new_columns], left_index=True, right_index=True, how='left')
        
        return X_tmp

    def transform(self, X):
        if not self.inference:
            X = self.create_train_features(X)
        return self.create_payload_features(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)