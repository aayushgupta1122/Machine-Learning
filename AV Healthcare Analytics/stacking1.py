# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:21:34 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:43:55 2020

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
info = pd.read_csv("train_data_dict.csv")
#np.shape(df)

X = df.iloc[:, 1:17].values
y = df.iloc[:, 17].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df = df.apply(le.fit_transform) #dataframe encoded directly
y = df.iloc[:, 17].values#.reshape(-1,1)
df = df.drop(['case_id', 'Stay'], axis = 1)
X = df.iloc[:, :].values
ctX = ColumnTransformer([('dfX', OneHotEncoder(), [1,3,5,6,7,11,12,14])], remainder = 'passthrough')
X = ctX.fit_transform(X)

cty = ColumnTransformer([('dfy', OneHotEncoder(), [0])], remainder = 'passthrough')
cty.fit(y.reshape(-1,1))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [47, 50]] = sc.fit_transform(X[:, [47, 50]])

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1) 
#y1 = y1.toarray()

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

base1 = GaussianNB()
base2 = RandomForestClassifier()
base3 = XGBClassifier()

#from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout

baseNN = Sequential()
baseNN.add(Dense(512, input_dim = 51, activation = 'relu', kernel_initializer = 'uniform'))
baseNN.add(Dropout(0.5))
baseNN.add(Dense(128, activation = 'relu', kernel_initializer = 'uniform'))
baseNN.add(Dropout(0.3))
baseNN.add(Dense(11, activation = 'softmax', kernel_initializer = 'uniform'))
baseNN.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

y_pred1 = np.empty((0,11),float)
y_pred2 = np.empty((0,11),float)
y_pred3 = np.empty((0,11),float)
y_predNN = np.empty((0,11),float)
y_test_fold = np.empty((0,11))

#from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 10)
for train_index, val_index in skf.split(X1, y1):
    X_train, X_val = X1[train_index], X1[val_index]
    y_train, y_val = y1[train_index], y1[val_index]
    
    y_train_NN = cty.transform(y_train.reshape(-1,1))
#    y_train = y_train.toarray()
#    y_val = y_val.toarray()

    base1.fit(X_train, y_train)
    y_pred1 = np.append(y_pred1, base1.predict_proba(X_val)).reshape(-1,11)
#    print("GNB" + str(accuracy_score(y_val, y_pred1)))
    print("GNB")
    
    base2.fit(X_train, y_train)
    y_pred2 = np.append(y_pred2, base2.predict_proba(X_val)).reshape(-1,11)
#    print("RF" + str(accuracy_score(y_val, y_pred2)))
    print("RF")
    
    base3.fit(X_train, y_train)
    y_pred3 = np.append(y_pred3, base3.predict_proba(X_val)).reshape(-1,11)
#    print("XGB" + str(accuracy_score(y_val, y_pred3)))
    print("XGB")
    
    baseNN.fit(X_train, y_train_NN.toarray(), epochs=25, batch_size=256)
    y_predNN = np.append(y_predNN, baseNN.predict(X_val)).reshape(-1,11)
#    print("NN" + str(accuracy_score(y_val, y_predNN)))
    print("NN")
    
    y_test_fold = np.append(y_test_fold, y_val.reshape(-1,1))
#    y_test_fold = cty.transform(y_test_fold)

meta_X = np.concatenate((y_pred1, y_pred2, y_pred3,y_predNN), axis = 1)
meta_y = y_test_fold
meta_test = np.concatenate((base1.predict_proba(X_test),base2.predict_proba(X_test),base3.predict_proba(X_test),baseNN.predict_proba(X_test)),axis=1)
  
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression()
meta.fit(meta_X, meta_y)
yhat = meta.predict(meta_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, yhat)  
    
    
    