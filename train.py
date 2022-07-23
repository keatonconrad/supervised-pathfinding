import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv("data_pathfinding_target.csv", sep="@").sample(frac=1)


def process(t):
    new = eval(
        t.replace("' '", "0")
        .replace("'s'", "1")
        .replace("'0'", "2")
        .replace("'1'", "3")
        .replace("'x'", "4")
    )
    return new


data["flat"] = [
    np.asarray([np.asarray(x) for x in process(state)]).astype("float32")
    for state in tqdm(data["state"])
]
# print(data['newstate'])
# data.drop('state', axis=1, inplace=True)
print(data["flat"])
print(
    data["flat"].iloc[0],
    data["flat"].iloc[0].shape,
    data["flat"].iloc[0].dtype,
    type(data["flat"].iloc[0][0]),
)
# data['flat'] = [np.asarray(state) for state in data['newstate']]
# data.drop('newstate', axis=1, inplace=True)

# x_data = pd.DataFrame(np.row_stack(data['flat'].tolist()))
x_data = pd.DataFrame()
x_data["flat"] = data["flat"]
x_data["target"] = data["target"]
print(x_data)
y_data = np.asarray(pd.get_dummies(data["label"])).astype("float32")
# print(y_data)

import tensorflow as tf
import keras
from keras.layers import Dense, Conv1D

state_input = keras.Input(shape=(6, 6))
x = Conv1D(32, 2, activation="relu")(state_input)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = tf.keras.layers.Flatten()(x)

target_input = keras.Input(shape=(1))
y = Dense(512, activation="relu")(target_input)

concat = tf.keras.layers.Concatenate()([x, y])
z = Dense(512, activation="relu")(concat)
z = Dense(512, activation="relu")(z)
z = Dense(512, activation="relu")(z)
outputs = Dense(4, activation="softmax")(z)

model = keras.Model(inputs=[state_input, target_input], outputs=outputs)
model.summary()

adam = keras.optimizers.Adam(0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["acc"])

state_data = np.asarray(data["flat"].to_list())
target_data = np.asarray(data["target"].to_list())
print(state_data)

history = model.fit(
    [state_data, target_data],
    y_data,
    validation_split=0.2,
    epochs=40,
    batch_size=1024,
)
