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
            "model": LogisticRegression(random_state=42),
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
                "classifier__kernel": ['linear', 'rbf']
            }
        },
        {
            "name": "DecisionTreeClassifier",
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [2, 4, 6, 8, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },
        {
            "name": "RandomForestClassifier",
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [2, 4, 6, 8, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },
        {
            "name": "XGBoostClassifier",
            "model": XGBClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [2, 4, 6, 8, None],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__gamma": [0, 0.1, 0.2]
            }
        },
        {
            "name": "LGBMClassifier",
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__num_leaves": [15, 31, 63],
                "classifier__max_depth": [2, 4, 6, 8, -1],
                "classifier__reg_alpha": [0, 0.1, 0.2]
            }
        },
        {
            "name": "CatBoostClassifier",
            "model": CatBoostClassifier(random_state=42, verbose=0, logging_level='Silent'),
            "param_grid": {
                "classifier__depth": [2, 4, 6, 8],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__iterations": [100, 200, 300],
            }
        }
    ]