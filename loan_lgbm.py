# coding: utf-8

import csv

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedKFold


def preprocessing(df_train, df_predict):
    # drop columns that are missing too much or helpless
    df_train.drop(['ID', 'state', 'employment', 'type_of_application', 'months_since_last_delinq', 'extended_reason'],
                  axis=1, inplace=True)
    df_predict.drop(
        ['ID', 'state', 'employment', 'type_of_application', 'months_since_last_delinq', 'loan_paid',
         'extended_reason'],
        axis=1, inplace=True)

    # fill missing data
    df_train['total_revolving_limit'] = df_train['total_revolving_limit'].fillna(
        df_train['total_revolving_limit'].mean())
    df_predict['total_revolving_limit'] = df_predict['total_revolving_limit'].fillna(
        df_predict['total_revolving_limit'].mean())

    # convert category data
    mapping_dict = {
        'employment_length': {'0': 0, '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
                              '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10},
    }
    df_train = df_train.replace(mapping_dict)
    df_predict = df_predict.replace(mapping_dict)

    for column in df_train.columns:
        col_type = df_train[column].dtype
        if col_type == 'object' or col_type.name == 'category':
            df_train[column] = df_train[column].astype('category')
            df_predict[column] = df_predict[column].astype('category')

    return df_train, df_predict


def run(train, test, num_folds, stratified=True):
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    X = train.drop(['loan_paid'], axis=1).copy()
    y = train['loan_paid'].copy()
    y_final = 0
    for n_fold, (train_index, test_index) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        parameters = {
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': 0.25,
            'learning_rate': 0.1,
            'zero_as_missing': True,
            'max_bin': 150,
            'feature_fraction': 0.5
        }

        # lgb fit train
        model = lgb.train(params=parameters, train_set=train_data, valid_sets=[test_data], num_boost_round=350,
                          verbose_eval=50)
        y_test_pred = model.predict(test)
        y_final += y_test_pred
        y_hat = model.predict(X_test)
        y_valid = (y_hat > 0.5) + 0
        print(y_valid)
        mcc = matthews_corrcoef(y_test, y_valid)
        print("%d Fold: %.6f" % (n_fold, mcc))

    y_final /= num_folds  # get average of kfold
    y_pred = (y_final > 0.5) + 0
    header = ['ID', 'loan_paid']

    with open('prediction_auc.csv', 'w', newline='') as fw:

        writer = csv.writer(fw)
        writer.writerow(header)
        for i, y in enumerate(y_pred):
            writer.writerow([i + 1000000, y])


if __name__ == "__main__":
    # read dataset
    df_train = pd.read_csv('lending_train.csv')
    df_predict = pd.read_csv('lending_topredict.csv')
    X_train, X_predict = preprocessing(df_train, df_predict)
    run(X_train, X_predict, 10, stratified=True)
