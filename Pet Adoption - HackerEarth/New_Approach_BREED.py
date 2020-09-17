# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:25:14 2020

@author: Admin
"""

import pandas as pd 
import numpy as np
import datetime 


df = pd.read_csv("train.csv").dropna()

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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
ct = ColumnTransformer([('color_type', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X.astype(float)



y = df.iloc[:,9].values

#df["height(cm)"] = df['height(cm)'] / 100
#df.rename(columns = {"height(cm)" : "height(m)"}, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)  


from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)

##Grid Search
#from sklearn.model_selection import GridSearchCV
#parameters = [
#        {'C' : np.logspace(-4,4,20),
#        'solver' : ['liblinear', 'saga', 'lbfgs', 'newton-cg']} 
#        ]
#grid_search = GridSearchCV(classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_score_gs = grid_search.best_score_
#best_parameters_gs = grid_search.best_params_



from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier, X = X_train, y =  y_train, cv = 10)
mean_kcv = kcv.mean()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
