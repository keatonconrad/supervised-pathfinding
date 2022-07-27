import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv('data_full_path.csv', sep='@').head(4000000)#.sample(frac=1)

def process(t):
  new = eval(t.replace("' '", "0").replace("'s'", "1").replace("'e'", "2").replace("'x'", "3"))
  return new

data['newstate'] = [process(state) for state in tqdm(data['state'])]
data.drop('state', axis=1, inplace=True)
data['flat'] = [np.asarray(state).flatten() for state in data['newstate']]
data.drop('newstate', axis=1, inplace=True)

x_data = pd.DataFrame(np.row_stack(data['flat'].tolist()))
y_data = np.array(pd.get_dummies(data['label'])).astype('float32')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=False)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

length = 3

train_grams = np.array([x_train.to_numpy()[i:i+length] for i in range(len(x_train.to_numpy())-length+1)])
test_grams = np.array([x_test.to_numpy()[i:i+length] for i in range(len(x_test.to_numpy())-length+1)])


model = Sequential()
model.add(keras.Input(shape=(length, len(x_data.columns))))
model.add(keras.layers.Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

adam = keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])


with tf.device('/device:GPU:0'):
    history = model.fit(train_grams, y_train[length-1:], validation_data=(test_grams, y_test[length-1:]), epochs=15, batch_size=1024)
    preds = model.predict(test_grams)

from sklearn.metrics import classification_report

preds = np.argmax(preds, axis=1)
y_test = np.argmax(y_test, axis=1)[length-1:]
print(classification_report(y_test, preds))
