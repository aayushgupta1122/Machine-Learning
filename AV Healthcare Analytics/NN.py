# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:29:00 2020

@author: Admin
"""

import numpy as np
import pandas as pd
from keras.utils import np_utils

df = pd.read_csv("train.csv")
sub = pd.read_csv("test.csv")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
df[['Bed Grade', 'City_Code_Patient']] = imputer.fit_transform(df[['Bed Grade', 'City_Code_Patient']])
sub[['Bed Grade', 'City_Code_Patient']] = imputer.transform(sub[['Bed Grade', 'City_Code_Patient']])
case_id = sub['case_id']
sub = sub.drop(['case_id'], axis = 1)

X = df.iloc[:, 1:17].values
y = df.iloc[:, 17].values
Test = sub.iloc[:,:].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_y = LabelEncoder()
#df = df.apply(le.fit_transform) #dataframe encoded directly
#y = df.iloc[:, 17].values.reshape(-1,1)
y = le_y.fit_transform(y)
y = np_utils.to_categorical(y)
df = df.drop(['case_id', 'Stay'], axis = 1)
#X = df.iloc[:, :].values
ctX = ColumnTransformer([('dfX', OneHotEncoder(), [1,3,5,6,7,11,12,14])], remainder = 'passthrough')
X = ctX.fit_transform(X)
Test = ctX.transform(Test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [47, 50]] = sc.fit_transform(X[:, [47, 50]])
Test[:, [47, 50]] = sc.transform(Test[:, [47, 50]])

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(512, input_dim = 51, activation = 'relu',  kernel_initializer = 'random_normal'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_initializer = 'random_normal'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_initializer = 'random_normal'))
#model.add(Dropout(0.2))
model.add(Dense(units=11, activation='softmax', kernel_initializer = 'random_normal'))

#from keras.optimizers import SGD
#opt = SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

model.fit(X, y, epochs=30, validation_split = 0.05, batch_size=256)
yhat = model.predict_classes(Test)

yhat = np.argmax(yhat, axis = 1)
stay = le_y.inverse_transform(yhat)

s = np.column_stack((case_id,stay))
s = pd.DataFrame(s)
s.columns = ['case_id', 'Stay']
s.to_csv("Submission.csv", index = False, index_label = None)    
