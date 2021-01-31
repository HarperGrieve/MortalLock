import keras as k
import tensorflow as tf


model = k.models.load_model('../Models/Trained-Model1612061950.5449553',
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