import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as K
from keras import backend as k
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from livelossplot import PlotLossesKeras
from datetime import datetime

# import keras.losses

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

current_time = str(time.time())

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../Models/Trained-Model' + current_time, save_best_only=True, monitor='val_loss',
                           mode='auto')

data = pd.read_excel('../Datasets/Full-Data-Set.xlsx')
output = data[['Home-Team-Win', 'Away-Team-Win', 'No-Bet', 'ML-Home', 'ML-Away']]

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
        no_bet = y_true[:, 2:3]
        odds_a = y_true[:, 3:4]
        odds_b = y_true[:, 4:5]
        # win_away_team = 1
        one = tf.constant(1, dtype=tf.float32)
        neg_one = tf.constant(1, dtype=tf.float32)
        # tf.cast(win_away_team, dtype=tf.float32)
        gain_loss_vector = k.concatenate([win_home_team * (odds_a - neg_one) + (one - win_home_team) * neg_one,
                                          win_away_team * (odds_b - neg_one) + (one - win_away_team) * neg_one,
                                          k.zeros_like(odds_a)], axis=1)
        loss = -1 * k.mean(k.sum(gain_loss_vector * y_pred, axis=1))
        return loss
    return loss


def get_model(input_dim, output_dim, base=1000, multiplier=0.25, p=0.2):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    l = tf.keras.layers.BatchNormalization()(inputs)
    l = tf.keras.layers.Dropout(p)(l)
    n = base
    l = tf.keras.layers.Dense(n, activation='relu')(l)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.Dropout(p)(l)
    n = int(n * multiplier)
    l = tf.keras.layers.Dense(n, activation='relu')(l)
    l = tf.keras.layers.BatchNormalization()(l)
    l = tf.keras.layers.Dropout(p)(l)
    n = int(n * multiplier)
    l = tf.keras.layers.Dense(n, activation='relu')(l)
    outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(l)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Nadam', loss=keras_custom_loss_function)
    return model


logdir = "../logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)


keras_custom_loss_function = keras_custom_loss_function()
# model = get_model(106, 3, 512, 0.5, 0.7)
# history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
#                     epochs=200, batch_size=8, callbacks=[EarlyStopping(patience=25), mcp_save,
#                                                          tensorboard_callback])


# print('Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y)))


print('done')

model = K.models.load_model('../Models/Trained-Model1612061950.5449553',
                            custom_objects={'loss': keras_custom_loss_function})
# print('Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y),
# model.evaluate(test_x, test_y)))

# test.drop(['Score', 'Home-Team-Win', 'Away-Team-Win',
# 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU',
#            'No-Bet', 'OU-Cover'], axis=1, inplace=True)
# print(data[16355:16356, :])
ml_predictions_array = [model.predict(data[16345:, :])]

for x in ml_predictions_array:
    print(x)
    print("-------------")
