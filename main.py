import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import time

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
import tensorflow as tf
import keras
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('NSE_TATA.csv')
#print(df.head())

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

data = df.sort_index(ascending=True)
new_dataset = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(df)):
    new_dataset['Close'][i] = data['Close'][i]
    new_dataset['Date'][i] = data['Date'][i]
    

#print(new_dataset.head())


new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
final_dataset=new_dataset.values
scaler=MinMaxScaler(feature_range=(0,1))
train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]
#final_dataset = [x[1] for x in final_dataset]
#print(final_dataset)

scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]
#print(scaled_data[60-60:60,0])
#print(scaled_data[60,0])
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

#print(inputs_data)

model = Sequential()
model.add(keras.Input(shape=(x_train_data.shape[1],1)))
model.add(keras.layers.LSTM(units=50,return_sequences=True))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dense(1))


model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.SparseTopKCategoricalAccuracy(), tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    

)



print("Fit model on training data")

history = model.fit(
        x_train_data,
        y_train_data,
        batch_size=1,
        epochs=10,
        verbose=2
        )


X_test=[]
y_test = []
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
    y_test.append(inputs_data[i,0])
X_test=np.array(X_test)
y_test=np.array(y_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_closing_price=model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

model.save("lstm_model.h5")

train_data=new_dataset[:987]
valid_data=new_dataset[987:]


valid_data['Predictions']=predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
