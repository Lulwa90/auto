#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:55:16 2018

@author: kevinschenthal

Analyze the model.

TODO:
    - (kevin.s): make into a jupyter notebook
"""

import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# LOAD DATA & MODEL
# =============================================================================

dir_data = '/Users/kevinschenthal/Desktop/kaggle/auto'
model_file = 'rf_clf.sav'

my_model = pd.read_pickle(os.path.join(dir_data, model_file))

# Get feature importances
for name, importance in zip(feature_names, my_model.feature_importances_):
    print(name, "=", importance)


