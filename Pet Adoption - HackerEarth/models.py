# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 07:45:48 2020

@author: Admin
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv").dropna()

df["height(cm)"] = df['height(cm)'] / 100

df.rename(columns = {"height(cm)" : "height(m)"}, inplace = True)

X = df.iloc[:,[3, 5, 6, 7, 8]].values
y = df.iloc[:,9].values


#df.head()
#
#df["pet_category"].unique()
#df["breed_category"].unique()
#df["color_type"].unique()
#df["condition"].unique()
#
#plt.scatter("pet_category", "breed_category", data = df)
#plt.scatter("pet_category", "color_type", data = df)
#plt.scatter("pet_category", "height(m)", data = df)
#plt.scatter("pet_category", "condition", data = df)
#plt.scatter("pet_category", "length(m)", data = df)
#plt.scatter("pet_category", "X1", data = df)
#plt.scatter("pet_category", "X2", data = df)

#Splitting data set into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)  

#Applying various classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 

models = []
models.append(("LR", LogisticRegression()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("NB", GaussianNB()))
models.append(("RF", RandomForestClassifier(n_estimators=100, criterion='gini')))

#K-fold Cross Validation
from sklearn.model_selection import cross_val_score

best_mean = []
best_std = []
names = []

model_performance = pd.DataFrame()

for name, model in models:
    kcv = cross_val_score(estimator = model, X = X_train, y =  y_train, cv = 10)
    best_mean.append((name, kcv.mean()))
    best_std.append((name, kcv.std()))
    model_performance = model_performance.append([kcv])
    names.append(name)
    
#Making predictions on Validation Set
predictions = pd.DataFrame()
for name, model in models:
    model.fit(X_train, y_train)
    predictions = predictions.append([model.predict(X_test)])
predictions = predictions.T
predictions.columns = [i for i in range(4)]

#Accuracy of Validation Set
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy = pd.DataFrame()
report = pd.DataFrame()
for i in range(4):
    y_pred = predictions[i]
    accuracy = accuracy.append([accuracy_score(y_test, y_pred)])
    report = report.append([classification_report(y_test, y_pred)])
    







