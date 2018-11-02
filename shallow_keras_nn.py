#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:36:18 2018

@author: kevinschenthal
"""

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
    - (kevin.s): gridsearch for hyperparameter tuning
    - (kevin.s): record results
"""

import os.path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
import tensorflow as tf

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
#data = df_test_over
#over_X = df_test_over[feature_names]
#over_y = df_test_over[target]
#data = train_X
#train_X = df_test_over[feature_names]
#over_y = df_test_over[target]

## Build model
#my_model = RandomForestClassifier(min_samples_split=4,
#                                  min_samples_leaf=2,
#                                  n_estimators=10,
#                                  max_depth=10,
#                                  class_weight='balanced',
#                                  random_state=0).fit(train_X, train_y)
from keras import models
from keras import layers
from keras import backend as K

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(train_X,
          train_y,
          validation_data=(val_X, val_y),
          callbacks=[roc_callback(training_data=(train_X, train_y),
                                  validation_data=(val_X, val_y))])

#history = model.fit(partial_x_train,
#                    partial_y_train,
#                    epochs=10,
#                    batch_size=512,
#                    validation_data=(x_val, y_val))

# =============================================================================
# PLOT LOSS OF TRAINING & VALIDATION
# =============================================================================
sys.exit()
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

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


# Get feature importances
#for name, importance in zip(feature_names, my_model.feature_importances_):
#    print(name, "=", importance)
    
# =============================================================================
# COMPARE TEST AND VALIDATION PERFORMANCE
# =============================================================================
