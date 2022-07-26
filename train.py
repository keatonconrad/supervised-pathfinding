import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.layers import Dense, Conv1D

data = pd.read_csv("data_pathfinding_target10.csv", sep="@").sample(frac=1)


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

y_data = np.asarray(pd.get_dummies(data["label"])).astype("float32")

state_input = keras.Input(shape=(10, 10))
# x = Conv1D(96, 6, activation="relu")(state_input)
x = Dense(512, activation="relu")(state_input)
x = keras.layers.Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = tf.keras.layers.Flatten()(x)

target_input = keras.Input(shape=(1))
y = Dense(512, activation="relu")(target_input)

concat = tf.keras.layers.Concatenate()([x, y])
z = Dense(1024, activation="relu")(concat)
z = Dense(1024, activation="relu")(z)
z = Dense(1024, activation="relu")(z)
outputs = Dense(4, activation="softmax")(z)

model = keras.Model(inputs=[state_input, target_input], outputs=outputs)
model.summary()

adam = keras.optimizers.Adam(0.001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["acc"])

state_data = np.asarray(data["flat"].to_list())
target_data = np.asarray(data["target"].to_list())

history = model.fit(
    [state_data, target_data],
    y_data,
    validation_split=0.2,
    epochs=40,
    batch_size=1024,
)
