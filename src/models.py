import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.preprocessing import get_preprocessor

# for DNN
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout


def build_dnn(n_features: int):
    """
    Build a Deep Neural Network model

    The neural net has 4 hidden dense layers using
    relu as the activation function and 1 output layer
    for binary classification.

    Input:
    n_features: int  Number of features from X.shape[1]

    Output:
    model: obj  DNN model
    """
    # we use the Sequential API
    model = Sequential()
    model.add(InputLayer(shape=(n_features,)))  # we add the input layer
    model.add(Dense(units=64, activation='relu'))  # first hidden layer with 64 neurons
    model.add(Dropout(0.2))  # we add some dropout layers to prevent overfitting
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # we compile the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model



