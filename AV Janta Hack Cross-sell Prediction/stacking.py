# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 06:53:05 2020

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

#neg, pos = np.bincount(y)
#total = neg + pos
#w0 = (1/neg)*(total)/2
#w1 = (1/pos)*(total)/2
#weights = {0: w0, 1: w1}

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

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0, shuffle = True, stratify = y)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout 
#from sklearn.utils.class_weight import compute_class_weight

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
#      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

base1 = XGBClassifier(verbosity = 2)
base2 = RandomForestClassifier(verbose = 2)
base3 = GradientBoostingClassifier(verbose = 2)
base4 = LGBMClassifier(verbose = 2)

model = Sequential()
model.add(Dense(256, input_dim = 14, kernel_initializer = 'glorot_uniform', activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(128, kernel_initializer = 'glorot_uniform', activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(32, kernel_initializer = 'glorot_uniform', activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = "sigmoid"))

model.compile(
        tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'
        ),
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=METRICS)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max')

base1.fit(X_train, y_train)
ypred1 = base1.predict_proba(X_val)
print('XGB')      
    
base2.fit(X_train, y_train)
ypred2 =  base2.predict_proba(X_val)
print('RF')     
    
base3.fit(X_train, y_train)
ypred3 = base3.predict_proba(X_val)
print('GBM') 

base4.fit(X_train, y_train)
ypred4 = base4.predict_proba(X_val)
print('LGBM') 

model.fit(X_train, y_train, epochs=15, validation_split = 0.15, callbacks = [early_stopping], batch_size=256)#, class_weight=weights)
ypred5 = model.predict(X_val)
print('NN')

level2X = np.concatenate((ypred1, ypred2, ypred3, ypred4, ypred5), axis = 1)
meta_sub = np.concatenate((base1.predict_proba(X_sub), base2.predict_proba(X_sub), base3.predict_proba(X_sub), base4.predict_proba(X_sub), model.predict(X_sub)), axis = 1)
#level2_1 = LogisticRegression()
#level2_1.fit(level2X, y_val)
#yhat1 = level2_1.predict_proba(meta_sub)
#print('LR') 
#
level2_2 = XGBClassifier()
level2_2.fit(level2X, y_val)
yhat = level2_2.predict(meta_sub)
print('XGB_M')

#estimators = [('LR', LogisticRegression()), ('XGB_M', XGBClassifier())]

#from sklearn.ensemble import VotingClassifier
#vote = VotingClassifier(estimators, voting = 'soft', verbose = True)
#vote.fit(level2X, y_val)
#yhat = vote.predict(meta_sub) 


s = np.column_stack((case_id,yhat))
s = pd.DataFrame(s)
s.columns = ['id', 'Response']
s.to_csv("Submission.csv", index = False, index_label = None)  


