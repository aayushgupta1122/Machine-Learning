# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:43:51 2020

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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7) 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_features='sqrt',bootstrap=True,min_samples_leaf=2,min_samples_split=5,criterion='entropy')
#classifier.fit(X_train, y_train)
#parameters = classifier.get_params()
#y_pred = classifier.predict(X_test)
#
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, y_pred)

#from sklearn.model_selection import cross_val_score
#kcv = cross_val_score(estimator = classifier, X = X_train, y =  y_train, cv = 10)
#best_mean = kcv.mean()
#best_std = kcv.std()


parameters = {
## 'bootstrap': [True, False],
## 'max_depth': [10, 20, 50, 80, 100, None],
## 'max_features': ['auto', 'sqrt'],
# 'criterion': ['gini', 'entropy'],
 'n_estimators': [400,500,600]
}
#
#Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_score_gs = grid_search.best_score_
best_parameters = grid_search.best_params_


