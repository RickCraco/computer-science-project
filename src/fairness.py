import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import equal_opportunity_difference

def evaluate_fairness(result_dict: dict, test_df: pd.DataFrame):
    """
    Perform fairness analysis on both synthetic and real model,
    it calculates metrics like Demographic Parity, Equalized Odds,
    and Equal Opportunity.

    Input:
    result_dict: dict  dictionary containing best_estimator syn and real pipelines
    test_df:  pd.DataFrame   test set of real data including the target column
    """
    # we take the pipeline from the result_dict (both syn and real)
    model_syn = result_dict["best_estimator_syn"]
    model_real = result_dict["best_estimator_real"]

    # we get the target column and the sensitive feature (Sex column)
    X_test = test_df.drop("HeartDisease", axis=1)  # we retrieve the feature test set for model prediction
    y_true = test_df["HeartDisease"]
    sensitive_fet = test_df["Sex"]

    # we make models predictions (syn and real)
    y_pred_syn = model_syn.predict(X_test)
    y_pred_real = model_real.predict(X_test)

    # we calculate the Demographic Parity metric
    print("\n ==== Demographic Parity (SYN) ====")
    print(demographic_parity_difference(y_true=y_true, y_pred=y_pred_syn, sensitive_features=sensitive_fet))
    
    print("\n ==== Demographic Parity (REAL) ====")
    print(demographic_parity_difference(y_true=y_true, y_pred=y_pred_real, sensitive_features=sensitive_fet))

    # we calculate the Equalized Odds metric
    print("\n ==== Equalized Odds (SYN) ====")
    print(equalized_odds_difference(y_true=y_true, y_pred=y_pred_syn, sensitive_features=sensitive_fet))

    print("\n ==== Equalized Odds (REAL) ====")
    print(equalized_odds_difference(y_true=y_true, y_pred=y_pred_real, sensitive_features=sensitive_fet))

    # we calculate the Equal Opportunity metric
    print("\n ==== Equal Opportunity (SYN) ====")
    print(equal_opportunity_difference(y_true=y_true, y_pred=y_pred_syn, sensitive_features=sensitive_fet))

    print("\n ==== Equal Opportunity (REAL) ====")
    print(equal_opportunity_difference(y_true=y_true, y_pred=y_pred_real, sensitive_features=sensitive_fet))