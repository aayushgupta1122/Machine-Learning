# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:01:20 2020

@author: Admin
"""

import numpy as np
import pandas as pd
import datetime
df = pd.read_csv("train.csv")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')


#df["height(cm)"] = df['height(cm)'] / 100
#df.rename(columns = {"height(cm)" : "height(m)"}, inplace = True)

#To include Turn Around time to the list of independent variables
format = '%Y-%m-%d %H:%M:%S'
df['issue_date'] = [datetime.datetime.strptime(x, format) for x in df['issue_date']]
df['listing_date'] = [datetime.datetime.strptime(x, format) for x in df['listing_date']]

t = pd.DataFrame()
t['TA Time'] = df['listing_date'] - df['issue_date']
t['TA Time'] = t['TA Time']/np.timedelta64(1, 'D')

c = df['color_type']

X = df.iloc[:,[3, 4, 5, 6, 7, 8]].values
t = t.iloc[:].values
X = np.append(X, t, axis = 1)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
ct = ColumnTransformer([('color_type', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X.astype(float)

y = df.iloc[:,9].values

from sklearn.model_selection import train_test_split
Train, Test, y_Train, y_Test = train_test_split(X, y, train_size = 0.7)  

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5)
for train_index, test_index in skf.split(Train, y_Train):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
from sklearn.ensemble import RandomForestClassifier
base_classifier1 = RandomForestClassifier(n_estimators=100)
base_classifier1.fit(X_train, y_train)
#parameters = classifier.get_params()
y_pred_1 = base_classifier1.predict(X_test)


from sklearn.linear_model import LogisticRegression
base_classifier2 = LogisticRegression()
base_classifier2.fit(X_train, y_train)
y_pred_2 = base_classifier2.predict(X_test)

#from sklearn.naive_bayes import GaussianNB
#base_classifier2 = GaussianNB()
#base_classifier2.fit(X_train, y_train)
#y_pred_2 = base_classifier2(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, y_pred_1)
accuracy2 = accuracy_score(y_test, y_pred_2)


meta_train_X = np.concatenate((y_pred_1.reshape(-1, 1), y_pred_2.reshape(-1, 1)), axis = 1)
meta_train_y = y_test
meta_test = np.concatenate((base_classifier1.predict(Test).reshape(-1,1), base_classifier2.predict(Test).reshape(-1,1)), axis = 1)

meta_classifier = LogisticRegression()
meta_classifier.fit(meta_train_X, meta_train_y)
yhat = meta_classifier.predict(meta_test)

final_accuracy = accuracy_score(y_Test, yhat)


#from sklearn.model_selection import cross_val_score
#kcv = cross_val_score(estimator = classifier, X = X_train, y =  y_train, cv = 10)
#best_mean = kcv.mean()
#best_std = kcv.std()


