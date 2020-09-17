# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:43:51 2020

@author: Admin
"""

import numpy as np
import pandas as pd
import datetime
df = pd.read_csv("train.csv")
sub = pd.read_csv("test.csv")

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

y = df.iloc[:,9].values #for breed_category

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7) 

format = '%Y-%m-%d %H:%M:%S'
sub['issue_date'] = [datetime.datetime.strptime(x, format) for x in sub['issue_date']]
sub['listing_date'] = [datetime.datetime.strptime(x, format) for x in sub['listing_date']]

t = pd.DataFrame()
t['TA Time'] = sub['listing_date'] - sub['issue_date']
t['TA Time'] = t['TA Time']/np.timedelta64(1, 'D')

X_Test = sub.iloc[:,[ 3,4,5,6,7,8]].values
t = t.iloc[:].values
X_Test = np.append(X_Test, t, axis = 1)
X_Test[:, 0:1] = imputer.transform(X_Test[:, 0:1])
X_Test[:, 1] = le.transform(X_Test[:, 1])
X_Test = ct.transform(X_Test)
X_Test = X_Test.astype(float)

#for Breed_Category
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred_Breed = classifier.predict(X_Test)


#for Pet_Category
y = df.iloc[:,10].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7) 

from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(C = 0.004832930238571752, solver = 'lbfgs', multi_class = 'multinomial')
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train, y_train)

y_pred_Pet = classifier.predict(X_Test)


submit = np.column_stack((sub['pet_id'].values, y_pred_Breed, y_pred_Pet))
submit = pd.DataFrame(submit)
submit.columns = ['pet_id', 'breed_category', 'pet_category']
submit.to_csv('submission.csv', index = False, index_label = None)



