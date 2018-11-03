#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:56:28 2018

@author: kevinschenthal

Analyze Data
"""

import os.path
import pandas as pd
import seaborn as sns

# Load the data
dir_data = '/Users/kevinschenthal/Desktop/kaggle/auto'
train = pd.read_csv(os.path.join(dir_data, 'training.csv'))
test = pd.read_csv(os.path.join(dir_data, 'test.csv'))

# =============================================================================
# GET CORRELATION PLOT & 
# =============================================================================

# Build a correlation matrix heatmap
corr_df = train.corr()
h = sns.heatmap(data=corr_df)
fig = h.get_figure()
fig.savefig(os.path.join(dir_data, 'correlation_heatmap.png'))

# Get paneplot
g = sns.pairplot(data=train)
fig2 = g.get_figure()
fig2.savefig(os.path.join(dir_data, 'auto_pairplot.png'))