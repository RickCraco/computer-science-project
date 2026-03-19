import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, log_loss

def plot_confusion_matrix(y_true, y_pred, labels= ["Negative", "Positive"], show_precision_recall=True):
    """
    Plot the confusion matrix of a trained model

    Input:
    y_true:   ground truth contains real classes
    y_pred:   models predictions
    labels: list   labels for class 0 and class 1 (0 = Negative, 1 = Positive)
    show_precision_recall: bool  if true plots values for precision and recall 
    """
    # we create the confusion matrix
    cm = confusion_matrix(y_true, y_pred) # in order tn, fp, fn, tp

    df_cm = pd.DataFrame(cm, index=labels, columns=["Predicted" + labels[0], "Predicted" + labels[0]])
    sns.heatmap(df_cm, annot=True, fmt='g')

    if show_precision_recall:
        plt.text(0, -0.1, f"Precision: {(cm[1][1]/(cm[1][1]+cm[0][1])):.3f}")
        plt.text(1, -0.1, f"Recall: {(cm[1][1]/(cm[1][1]+cm[1][0])):.3f}")


def evaluate_models(results_list: list, X_test_real: pd.DataFrame, y_test_real: pd.Series):
    """
    Evaluation of trained models comparing synthetic vs real performance.

    Input:
    results_list: list  List of dictionaries containing trained models using GridSearchCV
    X_test_real: pd.DataFrame  test set features
    y_test_real:  pd.Series    test set target column 
    """
    # we use a loop to go through each trained model
    for result in results_list:
        # we retrieve the model name
        model_name = result['model']

        # we retrieve the best estimator from GridSearchCV
        model_syn = result['best_estimator_syn']  # best estimator synthetic
        model_real = result['best_estimator_real']  # best estimator real 

        print("\n" + "="*40)
        print(f"Evaluating Model: {model_name}")

        # we make model predictions to evaluate metrics
        y_pred_syn = model_syn.predict(X_test_real)
        y_pred_real = model_real.predict(X_test_real)

        # we print the classification report for both synthetic and real
        # following the TSTR standard 
        print("\n ==== SYNTHETIC MODEL (TSTR) ====")
        print(classification_report(y_test_real, y_pred_syn))

        print("\n ==== REAL MODEL (TRTR) ====")
        print(classification_report(y_test_real, y_pred_real))