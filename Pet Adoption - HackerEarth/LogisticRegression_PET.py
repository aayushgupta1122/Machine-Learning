# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:28:20 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import datetime 

df = pd.read_csv("train.csv")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

#To include Turn Around time to the list of independent variables
format = '%Y-%m-%d %H:%M:%S' #Format of the timestamp in the dataset
df['issue_date'] = [datetime.datetime.strptime(x, format) for x in df['issue_date']] #Converting Strings to datetime datatype
df['listing_date'] = [datetime.datetime.strptime(x, format) for x in df['listing_date']]

t = pd.DataFrame()
t['TA Time'] = df['listing_date'] - df['issue_date']
t['TA Time'] = t['TA Time']/np.timedelta64(1, 'D')

#Creating the Independent and Dependent variables
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

y = df.iloc[:,10].values


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.7)  

#y_pred = pd.DataFrame()
#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits = 5)
#for train_index, test_index in skf.splits(X, y):
#    X_train, X_val = X[train_index], X[test_index]
#    y_train, y_val = y[train_index], y[test_index]
#    
#    classifier.fit(X_train, y_train)
#    y_pred1 = pd.DataFrame(classifier.predict(X_val))
#    y_pred = pd.concat([y_pred, y_pred1])


#from sklearn.linear_model import LogisticRegression
##classifier = LogisticRegression(C = 0.004832930238571752, solver = 'lbfgs', multi_class = 'multinomial')
#classifier = LogisticRegression(solver = 'newton-cg', C = 12)
#classifier.fit(X_train, y_train)
    
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.075)    
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_val)
#
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
##y_pred = classifier.predict(X_val)
#cm = confusion_matrix(y_val, y_pred)
#accuracy = accuracy_score(y_val, y_pred)

#Grid Search
#from sklearn.model_selection import GridSearchCV
#parameters = {
#        'n_estimators' : [200, 400]
#} 
#        
#grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
#grid_search = grid_search.fit(X_train, y_train)
#best_score = grid_search.best_score_
#best_parameters = grid_search.best_params_

model_performance = pd.DataFrame()

from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier, X = X_train, y =  y_train, cv = 10)
model_performance.append([kcv])
mean_kcv = kcv.mean()
std_kcv = kcv.std()

classifier.get_params()





