# from google.colab import drive
# drive.mount('/content/drive')

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
data = pd.read_csv('/content/drive/My Drive/COVID19/COVID19.csv')

data['Country'] = data['Country/Region']
ser=pd.Series(data=data[data['Province/State'].notnull()]['Country'].values+" "+data[data['Province/State'].notnull()]['Province/State'].values,index=data[data['Province/State'].notnull()]['Country'].index)
index= data[data['Province/State'].notnull()]['Country'].index

for i in index:
  data.at[i,"Country"]=ser[i]
data=data.set_index("Country")
data=data.drop(['Province/State', 'Country/Region', 'Lat', 'Long'],axis=1).T

data['pred']=data.India.shift(-1)
data=data.dropna()

values=data.values
values=values.astype('float32')


def scale(train, test):
	scaler = StandardScaler()
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def invert_scale(trainX,testX,scaler,model):
  X=np.concatenate((trainX,testX),axis=0)
  yhat=model.predict(X)
  data_pred=np.concatenate(((X.reshape(dataset[:,0:-1].shape)),yhat),axis=1)
  yf=scaler.inverse_transform(data_pred)[:,-1]
  return yf

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
dataset = values[:,[131,-1]]
# dataset=dataset[dataset>0]
dataset=dataset[dataset[:,0]>0]

plt.plot(dataset[:,0])
plt.show()

n=int(len(dataset)*0.90)
train,test=dataset[:n,:],dataset[n:,:]
scaler,train_scaled,test_scaled=scale(train,test)
trainX = train_scaled[:,0:-1]
trainY = train_scaled[:,-1]
#testX = dataX[int(0.67*len(dataX)):,:]
testX = test_scaled[:,0:-1]
testY = test_scaled[:,-1]
trainX = np.reshape(trainX, (trainX.shape[0], 1, 1))
testX = np.reshape(testX, (testX.shape[0], 1, 1))
# get the X_test from test and use this yhat to transform using the scaler obtained above and invert_scale function

look_back = 1

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Sequential
def get_model():
  model = Sequential()
  #model.add(LSTM(4, input_shape=(1, 256)))
  model.add(LSTM(350, input_shape=(1, 1)))
  model.add(Dense(1))
  # model.add(Dense(1,activation='exponential'))
  return model
model = get_model()
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2,validation_split=0.20,metrics=['val_loss'])
yf=invert_scale(trainX,testX,scaler,model)

fig, ax = plt.subplots()
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="validation loss")
plt.legend()
plt.show()

y=dataset[:,-1]
fig, ax = plt.subplots()
plt.plot(y,label="actual")
plt.plot(yf,label="predicted")
plt.legend()
fig.show()
