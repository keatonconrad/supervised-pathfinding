import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv", sep="@")  # .head(4000000)#.sample(frac=1)

data["flat"] = [
    np.asarray(ast.literal_eval(state)).flatten() for state in tqdm(data["state"])
]

x_data = pd.DataFrame(np.row_stack(data["flat"].tolist()))
x_data["curiosity"] = data["curiosity"]
print(x_data)
y_data = np.array(pd.get_dummies(data["label"])).astype("float32")
print(y_data)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, shuffle=False
)

length = 4

train_grams = np.array(
    [x_train[i : i + length] for i in tqdm(range(len(x_train) - length + 1))]
)
test_grams = np.array(
    [x_test[i : i + length] for i in tqdm(range(len(x_test) - length + 1))]
)


model = Sequential()
model.add(keras.Input(shape=(length, len(x_data.columns))))
model.add(keras.layers.Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(Dense(4, activation="softmax"))

adam = keras.optimizers.Adam(0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["acc"])


with tf.device("/device:GPU:0"):
    history = model.fit(
        train_grams,
        y_train[length - 1 :],
        validation_data=(test_grams, y_test[length - 1 :]),
        epochs=25,
        batch_size=1024,
    )
    preds = model.predict(test_grams)

from sklearn.metrics import classification_report

preds = np.argmax(preds, axis=1)
y_test = np.argmax(y_test, axis=1)[length - 1 :]
print(classification_report(y_test, preds))
