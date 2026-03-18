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