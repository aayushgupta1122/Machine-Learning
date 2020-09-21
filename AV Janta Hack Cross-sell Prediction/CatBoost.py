# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:52:42 2020

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')
df_sub = pd.read_csv('test.csv')
case_id = df_sub.id
df_sub = df_sub.drop(['id'], axis = 1)

#col_1=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
#cat_col = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
#X = df[col_1]
y = df['Response']
#X_sub = df_sub[col_1]
X = df.drop(['id','Response'], axis = 1)

cat_var = np.where(X.dtypes != np.float)[0]

neg, pos = np.bincount(y)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

over = RandomOverSampler(sampling_strategy = 0.4)
under = RandomUnderSampler(sampling_strategy = 0.8)
#smote = SMOTE(sampling_strategy = 0.4, random_state = 1)

#X, y = smote.fit_resample(X, y)
X, y = over.fit_resample(X,y)
X, y = under.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, shuffle = True, stratify = y)

from catboost import CatBoostClassifier 
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

classifier = CatBoostClassifier()#, scale_pos_weight = neg/pos)
classifier.fit(X_train, y_train, eval_set = (X_val, y_val), cat_features = cat_var)
yhat = classifier.predict(df_sub)

s = np.column_stack((case_id,yhat))
s = pd.DataFrame(s)
s.columns = ['id', 'Response']
s.to_csv("Submission.csv", index = False, index_label = None)  








