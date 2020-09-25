# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 08:20:41 2020

@author: Admin
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('news.csv')
subdataset = dataset[['title','text']]
subdataset = np.array(subdataset)

#Cleaning the Texts
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus = []
for i in range(0, 6335):
    review = re.sub('[^a-zA-Z]', ' ', str(subdataset[i, :]).strip('[]'))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2700)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 3].values

#We'll use Naive Bayes Classification
#Splitting dataset into Test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state = 0)

#Fitting Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting on validation set
y_pred = classifier.predict(X_test)

#Accuracy Score and Confusion Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#K-Fold Validation
from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
kcv.mean()
kcv.std()







