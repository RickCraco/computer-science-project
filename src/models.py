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

def full_pipeline(model_name: str, numeric_cols: list, categorical_cols: list):
    """
    Build full machine learning pipeline with preprocessing and model training

    Input:
    model_name: str  Name of the classifier
    numeric_cols: list   numeric features
    categorical_cols: list  categorical features

    Output:
    Pipeline: obj  sklearn Pipeline object
    """
    # we get the preprocessor
    preprocessor = get_preprocessor(numeric_features=numeric_cols, categorical_features=categorical_cols)

    # dictionary of all the models that will be tested
    models = {
        'logistic_regression': LogisticRegression(),
        'svm': SVC(probability=True),
        'decision_tree': DecisionTreeClassifier(criterion='gini'),
        'random_forest': RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4),
        'xgboost': XGBClassifier(eval_metric='logloss'),
        'lightgbm': LGBMClassifier(verbose=-1),
        'catboost':CatBoostClassifier(verbose=0),
        'dnn': KerasClassifier(model=build_dnn, epochs=50, verbose=0)
    }

    # check if the model is in the model dictionary
    if model_name not in models:
        raise ValueError(f"Model {model_name} not in models dictionary")

    # build the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models[model_name])
    ])

    return pipeline



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