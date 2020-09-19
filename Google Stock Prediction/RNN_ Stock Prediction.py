# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 07:17:14 2020

@author: Admin
"""

# Part 1 - Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the TRAINING Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values        #Keras only works with numpy arrays

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a Data Structure with 60 time steps and 1 output (preprocessing for RNN)
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])   #predicting future stock, one t+1
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #newshape = (batch_size, time_steps, input_dim)
    

# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor = Sequential()

#Adding the First LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1), dropout = 0.2))
#regressor.add(Dropout(rate = 0.2))

#Adding the Second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, dropout = 0.2))
#regressor.add(Dropout(rate = 0.2))

#Adding the Third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, dropout = 0.2))
#regressor.add(Dropout(rate = 0.2))

#Adding the Fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False, dropout = 0.2))
#regressor.add(Dropout(rate = 0.2))

#Adding the Output Layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making Predicitons and Visualising Results

#Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Google Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



