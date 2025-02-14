import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
import seaborn as sns
import pandas as pd
import lightgbm as lgbm

def objective(trial, X_train, y_train, eval_set):
    # Definição dos hiperparâmetros a serem ajustados
    params = {
        'verbose' : 0,
        'random_state' : 777,
        'early_stopping_rounds' : 5,
        'n_jobs' : -1,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    model = lgbm.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=eval_set, 
              eval_metric='auc')

    y_pred_valid = model.predict_proba(eval_set[0][0])[:, 1]
    y_pred_train = model.predict_proba(X_train)[:, 1]

    # Overfitting penalty
    roc_auc_valid = roc_auc_score(eval_set[0][1], y_pred_valid, max_fpr=0.01)
    penalty = abs(roc_auc_valid - roc_auc_score(y_train, y_pred_train, max_fpr=0.01))

    return roc_auc_valid - penalty

def apply_encoders(X, lst_encoders):
    X_tmp = X.reset_index(drop=True)
    for encoder in lst_encoders:
        X_tmp = encoder.transform(X_tmp)
    return X_tmp

def feature_stability_plot(X_train, X_test, features_to_analyse, method):

    if method == 'ks':
        df_ = pd.DataFrame([
            [feature, kstest(X_train[feature], X_test[feature])[0]] 
            for feature in features_to_analyse
        ], columns=['feature', 'ks_train_test']).sort_values(by='ks_train_test', ascending=False)

        plt.figure(figsize=(15, 5))

        sns.barplot(data=df_, x='ks_train_test', y='feature', color='orange')

        plt.xlabel('KS Treino x Validação', fontsize=12)
    
    else:
        df_ = pd.DataFrame([
            [feature, calculate_psi(X_train[feature], X_test[feature])] 
            for feature in features_to_analyse
        ], columns=['feature', 'psi_train_test']).sort_values(by='psi_train_test', ascending=False)

        plt.figure(figsize=(15, 5))

        sns.barplot(data=df_, x='psi_train_test', y='feature', color='orange')

        plt.xlabel('PSI Treino x Validação', fontsize=12)
    plt.ylabel('Variáveis', fontsize=12)
    plt.title('Estabilidade das Variáveis entre Treino e Validação', fontsize=14)

    plt.show()


# Credits to: mwburke
# https://github.com/mwburke/population-stability-index
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=1):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       mwburke.github.io.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)


def roi(X, y_true, y_pred, amount_col, max_fpr=1):
    """ Calculate ROI: (tn_amount_col * 0.05) - fn_amount_col 

    Args:
        X (pd.DataFrame): Feature matrix containing the probabilities.
        y_true (pd.Series or np.array): True labels (0 or 1).
        y_pred (pd.Series or np.array): Predicted scores.
        amount_col (str): Column name in X that represents the monetary amount.
        max_fpr (float, optional): False positive rate to cutoff the dataset.
    
    Returns:
        float: The calculated ROI value.
    """
    
    fpr, _, thresholds = roc_curve(y_true, y_pred)
    threshold = thresholds[np.argmax(fpr >= max_fpr)] if any(fpr >= max_fpr) else 1.0
    
    y_pred_cutoff = (y_pred >= threshold).astype(int)
    
    
    # Calculate ROI
    roi_value = (tn * 0.05) - fn
    
    return roi_value
