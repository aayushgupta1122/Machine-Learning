# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:07:20 2020

@author: Admin
"""

import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
info = pd.read_csv("train_data_dict.csv")
np.shape(df)

np.shape(df.Hospital_code.unique())[0]
np.shape(df.Hospital_type_code.unique())[0]
np.shape(df.Hospital_region_code.unique())[0]
np.shape(df.Department.unique())[0]
np.shape(df.Ward_Facility_Code.unique())[0]
np.shape(df.Stay.unique())[0]


#1 - 16, 17 (range for IV and DV)
#2,4,6,7,8,12,13,15 (IV that needs encoding)
#17 (DV needs encoding)

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

#cty = ColumnTransformer([('dfy', OneHotEncoder(), [0])], remainder = 'passthrough')
#y = cty.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.05, random_state = 1) 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score


models = [('LR', LogisticRegression()),('KNN', KNeighborsClassifier()), ('GNB', GaussianNB()),
          ('MNB', MultinomialNB()),('RF',RandomForestClassifier()),('XGB', XGBClassifier())]

accuracy = []
f1 = []

for name, model in models:
    print(name, end=" ")
    classifier = model
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    a=accuracy_score(y_val, y_pred)
    print(a)
    accuracy.append((name, a))
#    f1.append((name, f1_score(y_val, y_pred)))









