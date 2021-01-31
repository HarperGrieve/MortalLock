import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as k
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

current_time = str(time.time())

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
save = ModelCheckpoint('../Models/Trained-Model' + current_time, save_best_only=True, monitor='val_loss',
                       mode='auto')

data = pd.read_excel('../Datasets/Full-Data-Set.xlsx')
output = data[['Home-Team-Win', 'Away-Team-Win', 'ML-Home', 'ML-Away']]

data.drop(['Score', 'Home-Team-Win', 'Away-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU',
           'No-Bet', 'OU-Cover'], axis=1, inplace=True)

data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(output)
train_x, test_x, train_y, test_y, = train_test_split(x_train, y_train)


def keras_custom_loss_function():
    def loss(y_true, y_pred):
        win_home_team = y_true[:, 0:1]
        win_away_team = y_true[:, 1:2]

        odds_a = y_true[:, 2:3]
        odds_b = y_true[:, 3:4]

        one = tf.constant(1, dtype=tf.float32)
        neg_one = tf.constant(-1, dtype=tf.float32)

        gain_loss_vector = k.backend.concatenate([win_home_team * (odds_a - one) + (one - win_home_team) * neg_one,
                                                  win_away_team * (odds_b - one) + (one - win_away_team) * neg_one
                                                  ], axis=1)
        return -1 * k.backend.mean(k.backend.sum(gain_loss_vector * y_pred, axis=1))

    return loss


def get_model(input_dim, output_dim, base=1000, multiplier=0.25, rate=0.2):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    layer = tf.keras.layers.BatchNormalization()(inputs)
    layer = tf.keras.layers.Dropout(rate)(layer)
    nodes = base
    layer = tf.keras.layers.Dense(nodes, activation='relu')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(rate)(layer)
    nodes = int(nodes * multiplier)
    layer = tf.keras.layers.Dense(nodes, activation='relu')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(rate)(layer)
    nodes = int(nodes * multiplier)
    layer = tf.keras.layers.Dense(nodes, activation='relu')(layer)
    outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(layer)
    net = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    net.compile(optimizer='Nadam', loss=keras_custom_loss_function)
    return net


logdir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=logdir)

keras_custom_loss_function = keras_custom_loss_function()
model = get_model(106, 2, 512, 0.5, 0.7)
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=200, batch_size=8, callbacks=[early_stopping, save,
                                                         tensorboard_callback])

print(
    'Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y)))

print('done')
