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
from sklearn.metrics import r2_score

# Keep same random seed to get reproducible results || When ran on GPU results not reproducible

from tensorflow import set_random_seed
set_random_seed(1)
data = pd.read_csv('COVID13_04.csv')

data['Country'] = data['Country/Region']
ser=pd.Series(data=data[data['Province/State'].notnull()]['Country'].values+" "+data[data['Province/State'].notnull()]['Province/State'].values,index=data[data['Province/State'].notnull()]['Country'].index)
index= data[data['Province/State'].notnull()]['Country'].index

for i in index:
  data.at[i,"Country"]=ser[i]
data=data.set_index("Country")
data=data.drop(['Province/State', 'Country/Region', 'Lat', 'Long','Recovered','Deaths','Date'],axis=1).T

dataset=data["India"].values.reshape((-1,1))
dataset=dataset.reshape((len(dataset),1))[-30:]
data= pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
dataset=data[data.Status=='Confirmed']['TT'].cumsum().values

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

lst=[42,41,39,38]
yH=np.zeros(y.shape)
for iter0 in range(0,5):
    np.random.seed(i)
#    dataset=data["India"].values.reshape((-1,1))
    sequence= dataset
    n_steps=4
    n=lst[n_steps-1]
    n_features=1    
    scalerY= MinMaxScaler()
    scalerY.fit_transform(sequence[:n].reshape((n,1)))
    sequence = scalerY.transform(sequence.reshape((len(sequence),1)))
    X,y=split_sequence(sequence,n_steps)
    X_=X[:n,:]
    y_=y[:n]
    
    X_train=X[:n].reshape((n,n_steps,1))
    X_test=X[n:].reshape((len(X)-n,n_steps,1))
    
    y_train=y[:n].reshape((n,1))
    y_test=y[n:].reshape((len(X)-n,1))
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D,MaxPooling1D,Conv2D,Activation,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout,GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential()
    model.add(Conv1D(filters=32, strides=1,kernel_size=5,padding='same', activation='relu', input_shape=(n_steps, n_features)))
    model.add(Conv1D(filters=32, strides=1,kernel_size=2,padding='same', activation='relu'))
    model.add(Conv1D(filters=64, strides=1,kernel_size=2,padding='same', activation='relu'))
    model.add(Conv1D(filters=64, strides=1,kernel_size=2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    from keras.callbacks import EarlyStopping
    es=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.10, epochs = 200,batch_size=1)
    x=X_test[-1]
    if iter0==0:
        yF=make_data(x,model,20,scalerY)/5
        yH=scalerY.inverse_transform(model.predict(X.reshape((len(X),n_steps,n_features))).reshape((len(X),1)))/5
    else:
        yF+=make_data(x,model,20,scalerY)/5
        yH+=scalerY.inverse_transform(model.predict(X.reshape((len(X),n_steps,n_features))).reshape((len(X),1)))/5

from sklearn.metrics import r2_score
ya=scalerY.inverse_transform(y.reshape((len(y),1)))
r2_train=r2_score(ya[:n],yH[:n])
print(r2_train)
r2_test=r2_score(ya[n:],yH[n:])
print(r2_test)
fig, ax = plt.subplots( nrows=1, ncols=1)
plt.plot(yH,label='pred')
plt.plot(ya,label='actual')
plt.legend()
fig.savefig("pred_"+str(n_steps)+"_"+str(r2_test)+"_"+str(r2_train)+".png")



fig, ax = plt.subplots( nrows=1, ncols=1 )
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

#this functidaton takes an x as an input; in *transformed* form gives the next x in transformed form

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

x=X_test[-1]
#yF=make_data(x,model,20,scalerY)

y1=scalerY.inverse_transform(np.concatenate((y_train,y_test),axis=0))
y_fin=np.concatenate((yH,yF),axis=0)
plt.plot(y_fin,label='final')
plt.plot(y1,label='original')
plt.savefig("next_"+str(n_steps)+".png")
plt.legend()
