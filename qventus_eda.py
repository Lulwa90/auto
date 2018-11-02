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
    - (kevin.s): deal with imbalanced data
    - (kevin.s): encode categorical features
    - (kevin.s): gridsearch for hyperparameter tuning
"""

import os.path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

cols_with_nans = list(diagnostic_df.loc[diagnostic_df.fill_rates < 1., 'column'])
for col in cols_with_nans:
    # Train
    col_median = train[col].median()
    train[col] = train[col].fillna(col_median)
    
    # Test
    col_median = test[col].median()
    test[col] = test[col].fillna(col_median)

# =============================================================================
# STANDARDIZE NUMERIC VALUE RANGE
# =============================================================================


    
# =============================================================================
# GET CORRELATION PLOT & 
# =============================================================================

# Build a correlation matrix heatmap
#corr_df = train.corr()
#h = sns.heatmap(data=corr_df)
#fig = h.get_figure()
#fig.savefig(os.path.join(dir_data, 'correlation_heatmap.png'))

# Get paneplot
#g = sns.pairplot(data=train)
#fig2 = g.get_figure()
#fig2.savefig(os.path.join(dir_data, 'auto_pairplot.png'))

# =============================================================================
# SPECIFY AND SPLIT DATA
# =============================================================================

data = train
# Define features
excluded_cols = ['RefId', target]
feature_names = [i for i in data.columns
                 if i not in excluded_cols and (data[i].dtype in [np.int64, np.float64])]

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

print('Random over-sampling:')
print(df_test_over[target].value_counts())


# =============================================================================
# BUILD A QUICK MODEL
# =============================================================================

# Fit a fast model and examine feature importance
data = df_test_over
over_X = df_test_over[feature_names]
over_y = df_test_over[target]

# Build model
my_model = RandomForestClassifier(random_state=0).fit(over_X, over_y)

# =============================================================================
# TEST MODEL
# =============================================================================

from sklearn.metrics import classification_report
val_y_pred = my_model.predict(val_X)

class_report = classification_report(val_y, val_y_pred)

accuracy = sum(val_y == val_y_pred) / val_y.shape[0]

print(class_report)
print(accuracy)

# Get feature importances
for name, importance in zip(feature_names, my_model.feature_importances_):
    print(name, "=", importance)