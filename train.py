import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('data_pathfinding_target.csv', sep='@').sample(frac=.01)

def process(t):
  new = eval(t.replace("' '", "0").replace("'s'", "1").replace("'0'", "2").replace("'1'", "3").replace("'x'", "4"))
  return new

data['flat'] = [np.asarray([np.asarray(x) for x in process(state)]).astype('float32') for state in tqdm(data['state'])]
#print(data['newstate'])
#data.drop('state', axis=1, inplace=True)
print(data['flat'])
print(data['flat'].iloc[0], data['flat'].iloc[0].shape, data['flat'].iloc[0].dtype, type(data['flat'].iloc[0][0]))
#data['flat'] = [np.asarray(state) for state in data['newstate']]
#data.drop('newstate', axis=1, inplace=True)

#x_data = pd.DataFrame(np.row_stack(data['flat'].tolist()))
x_data = pd.DataFrame()
x_data['flat'] = data['flat']
x_data['target'] = data['target']
print(x_data)
y_data = np.array(pd.get_dummies(data['label'])).astype('float32')
print(y_data)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(keras.Input(shape=(None, 6,6)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
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
  history = model.fit(data['flat'].tolist(), y_data, validation_split=0.2, epochs=40, batch_size=1)