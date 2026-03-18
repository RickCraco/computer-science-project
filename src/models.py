from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# for DNN
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout

from src.preprocessing import get_preprocessor


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



def train_with_gridsearch(model_name: str, X_train, y_train, num_cols: list, cat_cols: list, param_grid: dict):
    """
    Trains a model using Gridsearch for hyperparameter tuning and cross-validation

    Input:
    model_name:  str  Classifier name
    X_train: np.Array   array of features for training
    y_train: np.Array   array containing the target class column
    num_cols: list  list of numeric features
    cat_cols: list  list of categorical features
    param_grid: dict  dictionary for hyperparameter tuning
    """
    # we get the pipeline
    pipeline = full_pipeline(model_name, num_cols, cat_cols)

    # we configure the Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # cross-validation folds
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # training of the model
    grid_search.fit(X_train, y_train)