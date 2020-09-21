# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:00:41 2020

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')
df_sub = pd.read_csv('test.csv')
case_id = df_sub.id
df_sub = df_sub.drop(['id'], axis = 1)

X = df.iloc[:, 1:11].values
y = df.iloc[:, 11].values
X_sub = df_sub.iloc[:, :].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder()
ctX = ColumnTransformer([('X', ohe, [0,5,6])], remainder='passthrough')
X = ctX.fit_transform(X)
X_sub = ctX.transform(X_sub)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X[:, [7,11,13]] = sc.fit_transform(X[:, [7,11,13]])
X_sub[:, [7,11,13]] = sc.transform(X_sub[:, [7,11,13]])

neg, pos = np.bincount(y)
total = neg + pos
w0 = (1/neg)*(total)/2
w1 = (1/pos)*(total)/2
#weights = {0: w0, 1: w1}
weights = [w0, w1]

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

over = RandomOverSampler(sampling_strategy = 0.4)
X, y = over.fit_resample(X,y)

#smote = SMOTE(sampling_strategy = 0.4, random_state = 1)
#X, y = smote.fit_resample(X, y)

under = RandomUnderSampler(sampling_strategy = 'majority')
X, y = under.fit_resample(X, y)

shuffler = np.random.permutation(len(X))
X = X[shuffler]
y = y[shuffler]

from lightgbm import LGBMClassifier 
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

classifier = LGBMClassifier(verbose=2, eval_class_weights = weights)
classifier.fit(X, y)
yhat = classifier.predict(X_sub)

s = np.column_stack((case_id,yhat))
s = pd.DataFrame(s)
s.columns = ['id', 'Response']
s.to_csv("Submission.csv", index = False, index_label = None)  








