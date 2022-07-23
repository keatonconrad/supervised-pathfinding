import pandas as pd
import numpy as np

data = pd.read_csv('data8x8.csv', sep='@').sample(frac=1)

def process(t):
  new = eval(t.replace("' '", "0").replace("'s'", "1").replace("'e'", "2").replace("'x'", "3"))
  return new

data['newstate'] = [process(state) for state in data['state']]
data.drop('state', axis=1, inplace=True)
data['flat'] = [np.asarray(state).flatten() for state in data['newstate']]
data.drop('newstate', axis=1, inplace=True)

x_data = pd.DataFrame(np.row_stack(data['flat'].tolist()))
#x_data['obedience'] = data['obedience']
print(x_data)
y_data = np.array(pd.get_dummies(data['label'])).astype('float32')
print(y_data)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(keras.Input(shape=(len(x_data.columns),)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

adam = keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

with tf.device('/device:GPU:0'):
  history = model.fit(x_data, y_data, validation_split=0.2, epochs=40, batch_size=1024)