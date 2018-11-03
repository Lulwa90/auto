#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:55:16 2018

@author: kevinschenthal

Analyze the model.
"""

import pandas as pd


# Get feature importances
for name, importance in zip(feature_names, my_model.feature_importances_):
    print(name, "=", importance)


