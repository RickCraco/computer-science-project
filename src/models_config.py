from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models_config():
    """
    Returns a list containing all the models that will
    be tested and their param_grid for gridsearch.

    Ouput: list  List of dictionaries, each dictionary is a model config
    """
    return [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(),
            "param_grid": {
                "classifier__penalty": ['l1','l2','elasticnet','none'], # regularization method
                "classifier__C": [0.01, 0.1, 1, 10, 100],   # inverse regularization parameter, also called penalty
                "classifier__solver": ['lbfgs','newton-cg','liblinear','sag','saga'],  # solver algorithm to optimise model performance
                "classifier__max_iter": [100, 500, 1000]  # number of iterations for model convergence 
            }
        },
        {
            "name": "SupportVectorClassifier",
            "model": SVC(),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__gamma": [0.001, 0.01, 0.1, 1],
                "classifier_kernel": ['linear', 'rbf']
            }
        }
    ]