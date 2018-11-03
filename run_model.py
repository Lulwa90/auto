#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:20:52 2018

@author: kevinschenthal

Resources:
    https://www.kaggle.com/c/DontGetKicked/data

TODO:
    - (kevin.s): handle the case when the test set has missing values in non-train
      cols
    - (kevin.s): determine parameter range for gridsearch
    - (kevin.s): performa a randomsearchcv
    - (kevin.s): determine why auc returns nan
    - (kevin.s): implement dimensionality reduction - use PCA
    - (kevin.s): count number of outliers in each column
    - (kevin.s): save model and move model analysis to a separate script
    - (kevin.s): extract additional features from SubModel like liters
    - (kevin.s): make labelencoding and one-hot-encoding a method
"""

import os.path
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import sys

# Machine learning methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
from sklearn import decomposition

from sklearn.model_selection import GridSearchCV

# Load the data
dir_data = '/Users/kevinschenthal/Desktop/kaggle/auto'
train = pd.read_csv(os.path.join(dir_data, 'training.csv'))
test = pd.read_csv(os.path.join(dir_data, 'test.csv'))

# Identify target column
target = 'IsBadBuy'

# =============================================================================
# Analyze column types, filles, and values
# =============================================================================

cols = []
fill_rates = []
n_unique = []
uniques = []
for col in train.columns:
    fill_rate = train[col].notnull().sum() / train.shape[0]
    n_uniq = train[col].nunique()
    col_uniq_rate = n_uniq / train.shape[0]
    cols.append(col)
    n_unique.append(n_uniq)
    fill_rates.append(fill_rate)
    uniques.append(col_uniq_rate)
    
diagnostic_df = pd.DataFrame({'column': cols,
                              'fill_rates': fill_rates,
                              'unique_rate': uniques,
                              'n_uniques': n_unique})
describe_df = (train.describe()
                    .transpose()
                    .reset_index()
                    .rename(columns={'index': 'column'}))
diagnostic_df = diagnostic_df.merge(describe_df, on='column')

# We note that the fill rate is at worse 0.95 for WheeltypeID, so we impute
# columns with the median

# Add column type
col_type_df = (train.dtypes.to_frame()
                           .reset_index()
                           .rename(columns={'index': 'column',
                                            0: 'dtype'}))
diagnostic_df = diagnostic_df.merge(col_type_df, on='column')
  
# =============================================================================
# IMPUTE MISSING VALUES
# =============================================================================
  

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

missing_cat_impute = {'TopThreeAmericanName': 'OTHER',  # given category
                      'Color': 'NOT AVAIL',  # given category
                      'PRIMEUNIT': 'NOT AVAIL',  # Mostly nan
                      'Size': 'MEDIUM',  # Most common 
                      'WheelType': 'random',  # tie between Alloy & Covers, decreases feature power
                      'WheelTypeID': 'random',
                      'Nationality': 'AMERICAN',  # Most common
                      'Transmission': 'AUTO',  # Most Common
                      'AUCGUART': 'GREEN',  # Most Common
                      }

def fill_null_with_random(df, column):
        null_rows = df[cat_col].isnull()
        #count rows with NaNs
        l = null_rows.sum()
        #create array with size l
        s = np.random.choice(df.loc[df[column].notnull(), column], size=l)
        #set NaNs values
        df.loc[null_rows, column] = s   

for cat_col in missing_cat_impute.keys():
    # Random fill with choice if unknown
    if missing_cat_impute[cat_col] == 'random':
        fill_null_with_random(train, cat_col)
        fill_null_with_random(test, cat_col)
    # Impute with chosen value
    else:
        train[cat_col] = train[cat_col].fillna(missing_cat_impute[cat_col])
        test[cat_col] = test[cat_col].fillna(missing_cat_impute[cat_col])

cat_cols = ['TopThreeAmericanName', 'VNST', 'Make',]
one_hot_train = pd.get_dummies(train[cat_cols])


cols_with_nans = list(diagnostic_df.loc[diagnostic_df.fill_rates < 1., 'column'])
for col in cols_with_nans:
    # Train
    col_median = train[col].median()
    train[col] = train[col].fillna(col_median)
    
    # Test
    col_median = test[col].median()
    test[col] = test[col].fillna(col_median)

# =============================================================================
# GET DATES & TIME FEATURES
# =============================================================================

def split_date_col(arg_df, input_dt_col, output_dt_cols=['year', 'month', 'day'],
                   separator='/', dt_format="%Y,%m,%d"):
    arg_df[input_dt_col] = pd.to_datetime(arg_df[input_dt_col])
    arg_df[output_dt_cols[0]] = arg_df[input_dt_col].dt.year
    arg_df[output_dt_cols[1]] = arg_df[input_dt_col].dt.month
    arg_df[output_dt_cols[2]] = arg_df[input_dt_col].dt.day

purch_cols = ['PurchYear', 'PurchMonth', 'PurchDay']
split_date_col(train, 'PurchDate', output_dt_cols=purch_cols)
split_date_col(test, 'PurchDate', output_dt_cols=purch_cols)
train = train.drop('PurchDate', 1)
test = test.drop('PurchDate', 1)

# =============================================================================
# LABEL ENCODE CATEGORICAL FEATURES
# =============================================================================

# Submodel encoding because it has too many values for one-hot encoding
label_encode_cols = (['Model', 'SubModel', 'Trim']
                    + ['Auction', 'Make', 'Color', 'Transmission', 'WheelTypeID',
                'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName',
                'PRIMEUNIT', 'AUCGUART', 'VNST'])
for col in label_encode_cols:
    # Train
    le = LabelEncoder()
    le.fit(train[col].fillna('a'))
    train[col] = le.transform(train[col].fillna('a'))
    # Test
    le = LabelEncoder()
    le.fit(test[col].fillna('a'))
    test[col] = le.transform(test[col].fillna('a'))

# =============================================================================
# STANDARDIZE NUMERIC VALUE RANGE
# =============================================================================

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

numeric_cols = list(set(diagnostic_df['column']) - {'RefId', 'IsBadBuy'})
numeric_cols = numeric_cols + label_encode_cols
train[numeric_cols] = min_max_scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = min_max_scaler.fit_transform(test[numeric_cols])


# =============================================================================
# ONE-HOT ENCODE CATEGORICAL FEATURES
# =============================================================================

# One hot encode columns
#one_hot_cols = ['Auction', 'Make', 'Color', 'Transmission', 'WheelTypeID',
#                'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName',
#                'PRIMEUNIT', 'AUCGUART', 'VNST']
#one_hot_train = pd.get_dummies(train[one_hot_cols])
#train = (train.drop(one_hot_cols, 1)
#              .merge(one_hot_train, left_index=True, right_index=True))
#one_hot_test = pd.get_dummies(test[one_hot_cols])
#test = (test.drop(one_hot_cols, 1)
#            .merge(one_hot_test, left_index=True, right_index=True))

# =============================================================================
# REDUCE DIMENSIONS
# =============================================================================

#from sklearn.decomposition import PCA
#pca = PCA(n_components=30)
#train_pca = pca.fit_transform(train)
#test = pca.fit_transform(test)

# =============================================================================
# SPECIFY AND SPLIT DATA
# =============================================================================

data = train
# Define features
excluded_cols = ['RefId', target]
feature_names = [i for i in data.columns if i not in excluded_cols]

t_train, t_val= train_test_split(train, random_state=1)

# Specify data and target
train_X = t_train[feature_names]
train_y = t_train[target]
val_X = t_val[feature_names]
val_y = t_val[target]   

# =============================================================================
# OVERSAMPLE ISBADBUY CLASS
# =============================================================================

count_class_0, count_class_1 = t_train[target].value_counts()

# Divide by class
df_class_0 = t_train[t_train[target] == 0]
df_class_1 = t_train[t_train[target] == 1]

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

#print('Random over-sampling:')
#print(df_test_over[target].value_counts())


# =============================================================================
# BUILD A QUICK MODEL
# =============================================================================

# Fit a fast model and examine feature importance
#data = df_test_over
#over_X = df_test_over[feature_names]
#over_y = df_test_over[target]
#data = train_X
#train_X = df_test_over[feature_names]
#over_y = df_test_over[target]

# Build model
my_model = RandomForestClassifier(min_samples_split=4,
                                  min_samples_leaf=2,
                                  n_estimators=20,
                                  max_depth=10,
                                  class_weight='balanced',
                                  random_state=0).fit(train_X, train_y)

param_grid = {'n_estimators': [10, 20, 30],
              'min_samples_split': [2, 4, 6],
              'min_samples_leaf' : [2, 4, 5],
              'max_depth': [9, 10, 11]
}
print("Running Gridsearch")
CV_rfc = GridSearchCV(estimator=my_model,
                      param_grid=param_grid,
                      cv=5,
                      scoring='roc_auc')
CV_rfc.fit(train_X, train_y)
print(CV_rfc.best_params_)

# =============================================================================
# COMPARE TEST AND VALIDATION PERFORMANCE
# =============================================================================

# Get Test set data results
train_y_pred = my_model.predict(train_X)
class_report_train = classification_report(train_y, train_y_pred)
accuracy_train = sum(train_y == train_y_pred) / train_y.shape[0]
fpr, tpr, thresholds = roc_curve(train_y, train_y_pred, pos_label=2)
auc_value_train = auc(fpr, tpr)
print(class_report_train)
print(accuracy_train)
print(auc_value_train)

# Get validation set data results
val_y_pred = my_model.predict(val_X)
class_report_val = classification_report(val_y, val_y_pred)
accuracy_val = sum(val_y == val_y_pred) / val_y.shape[0]
fpr, tpr, thresholds = roc_curve(val_y, val_y_pred, pos_label=2)
auc_value = auc(fpr, tpr)
print(class_report_val)
print(accuracy_val)
print(auc_value)

sys.exit()

# Get feature importances
for name, importance in zip(feature_names, my_model.feature_importances_):
    print(name, "=", importance)
    
# =============================================================================
# COMPARE TEST AND VALIDATION PERFORMANCE
# =============================================================================


# =============================================================================
# SAVE RESULTS & SUBMIT RESULTS IN KAGGLE FORMAT
# =============================================================================

save_model = True
if save_model:
    model_path = os.path.join(dir_data, 'rf_clf.sav')
    pickle.dump(my_model, open(model_path, 'wb'))

get_submission = False
if get_submission:
    test_X = test[feature_names]
    #test_y_pred = my_model.predict(test_X)
    # Get probability
    test_y_pred_prob = my_model.predict_proba(test_X)[:, 1]
    
    submission = pd.DataFrame({'RefId': test['RefId'],
                               'IsBadBuy': test_y_pred_prob})
    submission.to_csv(os.path.join(dir_data,'submission.csv'), index=False,
                      columns=['RefId', 'IsBadBuy'])

