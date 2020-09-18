# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:08:09 2020

@author: Admin
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv', nrows = 5000).dropna()
#subdataset = dataset[['text', 'selected_text']]
#subdataset = np.array(subdataset)

import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
corpus = []
for i in range(5000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['selected_text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 3].values

vocab = cv.vocabulary_
the_words = cv.get_feature_names()


#Splitting the dataset into Train and Validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

#Applying SVM
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting results of test set
y_pred = classifier.predict(X_test)

#Accuracy Score and Confusion Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#K-Fold Validation
from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
kcv.std()
kcv.mean()
    



