#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:10:12 2020

@author: divyarth
"""


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Keep same random seed to get reproducible results || When ran on GPU results not reproducible
np.random.seed(7)
data = pd.read_csv('COVID19_5apr.csv')

data['Country'] = data['Country/Region']
ser=pd.Series(data=data[data['Province/State'].notnull()]['Country'].values+" "+data[data['Province/State'].notnull()]['Province/State'].values,index=data[data['Province/State'].notnull()]['Country'].index)
index= data[data['Province/State'].notnull()]['Country'].index

for i in index:
  data.at[i,"Country"]=ser[i]
data=data.set_index("Country")
data=data.drop(['Province/State', 'Country/Region', 'Lat', 'Long'],axis=1).T

dataset=data["India"].value
dataset=dataset.reshape((len(dataset),1))

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps=4
n_features=1
n=int(len(data)*0.8)
X,y=split_sequence(sequence,n_steps)
X_=X[:n,:]
y_=y[:n]

scaler = MinMaxScaler()
scaler.fit_transform(X_)
X_train=scaler.transform(X_).reshape((n,n_steps,1))
X_test=scaler.transform(X[n:,:]).reshape((len(X)-n,n_steps,1))

scalerY= MinMaxScaler()
scalerY.fit_transform(y_.reshape((len(y_),1)))
y_train=scalerY.transform(y_.reshape((len(y_),1))).reshape(y_.shape)
y_test=scalerY.transform(y[n:].reshape((len(y[n:]),1))).reshape(y[n:].shape)

from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Conv2D,Activation,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout,GlobalAveragePooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model

history = model.fit(X_train, y_train, validation_split=0.10, epochs = 150 ,batch_size=1)
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#history = model.fit(X_train, y_train, validation_split=0.10, epochs = 150 ,batch_size=1)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',padding='causal',dilation_rate=1, input_shape=(n_steps, n_features)))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',padding='causal',dilation_rate=2,))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',padding='causal',dilation_rate=4,))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))




from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

yH=scalerY.inverse_transform(model.predict(scaler.transform(X).reshape((len(X),n_steps,n_features))))

plt.plot(yH,label='pred')
plt.plot(y,label='actual')
plt.legend()

'''


from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate

def get_model(input_shape=None,n_filters=32,filter_width=2,dilation_rates=[2**i for i in range(8)]):
    
    img_input=Input(input_shape)
    x=img_input
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=filter_width, 
                   padding='causal',
                   dilation_rate=dilation_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(1)(x)

    return Model(img_input,x)

input_shape=X[0].shape
n_filters = 32 
filter_width = 2
dilation_rates = [2**i for i in range(8)] 

model=get_model(input_shape=input_shape)

from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.10, epochs = 150 ,batch_size=3,callbacks=[es])

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

history_seq = Input(shape=X[0].shape)
x = history_seq

for dilation_rate in dilation_rates:
    x = Conv1D(filters=n_filters,
               kernel_size=filter_width, 
               padding='causal',
               dilation_rate=dilation_rate)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(.2)(x)
x = Dense(1)(x)

def slice(x, seq_length):
    return x[:,-seq_length:,:]

pred_seq_train = Lambda(slice, arguments={'seq_length':3})(x)

model = Model(history_seq, pred_seq_train)

'''
