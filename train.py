import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('data_full_path.csv', sep='@').sample(frac=0.01)

def process(t):
  new = eval(t.replace("' '", "0").replace("'s'", "1").replace("'e'", "2").replace("'x'", "3"))
  return new

data['newstate'] = [process(state) for state in tqdm(data['state'])]
data.drop('state', axis=1, inplace=True)
data['flat'] = [np.asarray(state).flatten() for state in data['newstate']]
data.drop('newstate', axis=1, inplace=True)

x_data = pd.DataFrame(np.row_stack(data['flat'].tolist()))
print(x_data)
y_data = np.array(pd.get_dummies(data['label'])).astype('float32')
print(y_data)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.preprocessing.sequence import TimeseriesGenerator

length = 4

train_gen = TimeseriesGenerator(x_train, y_train, length=length, batch_size=1024)
test_gen = TimeseriesGenerator(x_test, y_test, length=length, batch_size=1024)

model = Sequential()
model.add(keras.Input(shape=(length, len(x_data.columns))))
model.add(GRU(4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

adam = keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

#model.summary()

with tf.device('/device:GPU:0'):
  history = model.fit(train_gen, validation_data=test_gen, epochs=40)