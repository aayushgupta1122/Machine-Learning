# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 06:58:55 2020

@author: Admin
"""
#%matplotlib 

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt 

df = pd.read_csv("train.csv")
#df_t = pd.read_csv("train.csv").head(10000) 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder()
#df = df.apply(le.fit_transform)
X = df.iloc[:, 1:11].values
y = df.iloc[:, 11].values

ct_X = ColumnTransformer([('X', ohe, [0,5,6])], remainder='passthrough')
X = ct_X.fit_transform(X)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X[:, [7,11,13]] = sc.fit_transform(X[:, [7,11,13]])


#plt.scatter(df['Response'], df["Age"])
#plt.scatter(df['Previously_Insured'], df["Annual_Premium"])
#plt.scatter(df['Vintage'], df["Annual_Premium"])
#plt.scatter(df['Policy_Sales_Channel'], df["Annual_Premium"])
#plt.scatter(df['Vehicle_Damage'], df["Annual_Premium"])
#plt.scatter(df['Vehicle_Age'], df["Annual_Premium"])
#unique, counts = np.unique(df['Region_Code'], return_counts=True)
#plt.bar(unique, counts)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = [('LR', LogisticRegression()), ('KNN', KNeighborsClassifier()), ('GNB', GaussianNB()), 
          ('RF', RandomForestClassifier()), ('GBM', GradientBoostingClassifier()), ('XGB', XGBClassifier())]

from sklearn.metrics import accuracy_score, f1_score
accu = []
f1 = {}

for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accu.append((name, accuracy_score(y_test, pred)))
    f1[name] =  f1_score(y_test, pred)
    print(accu)
    print(f1)
    



