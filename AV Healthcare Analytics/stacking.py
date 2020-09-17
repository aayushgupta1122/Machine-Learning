# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:43:55 2020

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")

sub = pd.read_csv("test.csv")
case_id = sub['case_id']
sub = sub.drop(['case_id'], axis = 1)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
df[['Bed Grade', 'City_Code_Patient']] = imputer.fit_transform(df[['Bed Grade', 'City_Code_Patient']])
sub[['Bed Grade', 'City_Code_Patient']] = imputer.transform(sub[['Bed Grade', 'City_Code_Patient']])

X = df.iloc[:, 1:17].values
y = df.iloc[:, 17].values
Test = sub.iloc[:,:].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = df.drop(['case_id', 'Stay'], axis = 1)

le_y = LabelEncoder()
y = le_y.fit_transform(y)

ctX = ColumnTransformer([('dfX', OneHotEncoder(), [1,3,5,6,7,11,12,14])], remainder = 'passthrough')
X = ctX.fit_transform(X)
Test = ctX.transform(Test)

cty = ColumnTransformer([('dfy', OneHotEncoder(), [0])], remainder = 'passthrough')
cty.fit(y.reshape(-1,1))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [47, 50]] = sc.fit_transform(X[:, [47, 50]])
Test[:, [47, 50]] = sc.transform(Test[:, [47, 50]])

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

base1 = GaussianNB()
base2 = RandomForestClassifier()
#base3 = XGBClassifier()

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

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 10)
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    y_train_NN = cty.transform(y_train.reshape(-1,1))

    base1.fit(X_train, y_train)
    y_pred1 = np.append(y_pred1, base1.predict_proba(X_val)).reshape(-1,11)
    print("GNB")
    
    base2.fit(X_train, y_train)
    y_pred2 = np.append(y_pred2, base2.predict_proba(X_val)).reshape(-1,11)
    print("RF")
    
#    base3.fit(X_train, y_train)
#    y_pred3 = np.append(y_pred3, base3.predict_proba(X_val)).reshape(-1,11)
##    print("XGB" + str(accuracy_score(y_val, y_pred3)))
#    print("XGB")
    
    baseNN.fit(X_train, y_train_NN.toarray(), epochs=25, batch_size=256)
    y_predNN = np.append(y_predNN, baseNN.predict(X_val)).reshape(-1,11)
    print("NN")
    
    y_test_fold = np.append(y_test_fold, y_val.reshape(-1,1))


meta_X = np.concatenate((y_pred1, y_pred2,y_predNN), axis = 1)
meta_y = y_test_fold
meta_test = np.concatenate((base1.predict_proba(Test),base2.predict_proba(Test),baseNN.predict_proba(Test)),axis=1)
  
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(max_iter = 200)
meta.fit(meta_X, meta_y)
yhat = meta.predict(meta_test)

from xgboost import XGBClassifier
meta = XGBClassifier()
meta.fit(meta_X, meta_y)
yhat = meta.predict(meta_test)

#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, yhat)  

stay = le_y.inverse_transform(yhat.astype(int))


s = np.column_stack((case_id,stay))
s = pd.DataFrame(s)
s.columns = ['case_id', 'Stay']
s.to_csv("Submission.csv", index = False, index_label = None)    
    
    