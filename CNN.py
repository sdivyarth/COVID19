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

sequence= data["India"]
n_steps=4
n_features=1
n=int(len(data)*0.8)
scalerY= MinMaxScaler()
scalerY.fit_transform(sequence[:n].values.reshape((n,1)))
sequence = scalerY.transform(sequence.values.reshape((len(sequence),1)))
X,y=split_sequence(sequence,n_steps)
X_=X[:n,:]
y_=y[:n]


X_train=X[:n].reshape((n,n_steps,1))
X_test=X[n:].reshape((len(X)-n,n_steps,1))

y_train=y[:n].reshape((n,1))
y_test=y[n:].reshape((len(X)-n,1))

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
from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.10, epochs = 150 ,batch_size=1)



plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

yH=scalerY.inverse_transform(model.predict(X.reshape((len(X),n_steps,n_features))).reshape((len(X),1)))

plt.plot(yH,label='pred')
plt.plot(scalerY.inverse_transform(y.reshape((len(y),1))),label='actual')
plt.legend()

#this functidaton takes an x as an input; in *transformed* form gives the next x in transformed form
x=X_test[-1]
def make_data(x,model,iter0,scalerY):
    x_=x.reshape(1,x.shape[0],1)
    z=model.predict(x_)
    ret=np.concatenate((x_[:,1:,:],z.reshape((1,1,1))),axis=1)
    y=[ret[:,-1,:]]
    for i in range(iter0):
        z=model.predict(ret)
        ret=np.concatenate((ret[:,1:,:],z.reshape((1,1,1))),axis=1)
        y.append(ret[:,-1,:])
    return scalerY.inverse_transform(np.array(y).reshape((len(y),1)))
yF=make_data(x,model,15,scalerY)
y1=data["India"].values.reshape(len(data),1)
y_fin=np.concatenate((y1,yF),axis=0)
plt.plot(y_fin,label='final')
plt.plot(y1,label='original')
plt.legend()
