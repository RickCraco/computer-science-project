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

    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(cm, index=labels, columns=["Predicted" + labels[0], "Predicted" + labels[1]])
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax)

    if show_precision_recall:
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])

        ax.text(0.5, -0.2, f"Precision: {precision:.3f}   Recall: {recall:.3f}", ha='center', transform=ax.transAxes)

    plt.tight_layout()

    return fig


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

        # we plot the confusion matrix for both synthetic and real
        fig = plot_confusion_matrix(y_test_real, y_pred_syn)
        plt.title("Confusion Matrix - Synthetic")
        plt.show()

        fig = plot_confusion_matrix(y_test_real, y_pred_real)
        plt.title("Confusion Matrix - Real")
        plt.show()

        # we check the logloss values for both synthetic and real
        # condition to check wether the model has predict_proba method
        if hasattr(model_syn, "predict_proba") and hasattr(model_real, "predict_proba"):
            # we use predict proba to predict the probability instead of the class label
            y_proba_syn = model_syn.predict_proba(X_test_real)
            y_proba_real = model_real.predict_proba(X_test_real)

            # we calculate the log loss
            print("\n ==== LOG LOSS ====")
            print(f"Synthetic loss: {log_loss(y_test_real, y_proba_syn):.4f}")
            print(f"Real loss: {log_loss(y_test_real, y_proba_real):.4f}")

        print("\n ==== ROC CURVE ====")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model_syn, X_test_real, y_test_real, ax=ax) # we plot the ROC Curve and calculate the AUC score
        ax.set_title("ROC Curve Synthetic")
        plt.show()

        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model_real, X_test_real, y_test_real, ax=ax)
        ax.set_title("ROC Curve Real")
        plt.show()

        print("\n ==== CROSS-VALIDATION ANALYSIS ====")
        # we retrieve the GridSearch obj 
        grid_syn = result["grid_search_syn"]
        grid_real = result["grid_search_real"]

        # we create a df containing the cv_results_ dict
        cv_syn_df = pd.DataFrame(grid_syn.cv_results_)
        cv_real_df = pd.DataFrame(grid_real.cv_results_)

        # we calculate the mean accuracy score for each fold
        mean_cv_syn = cv_syn_df["mean_test_score"].mean()
        mean_cv_real = cv_real_df["mean_test_score"].mean()

        print(f"Mean CV score (Synthetic): {mean_cv_syn:.4f}")
        print(f"Mean CV score (Real):      {mean_cv_real:.4f}")
        print(f"Difference (Syn - Real):   {(mean_cv_syn - mean_cv_real):.4f}")



def plot_learning_curve(history, label: str):
    """
    Plot the learning curve for loss and accuracy
    for a Deep Neural Network model.

    Input:
    history: obj   History model containing the information of a trained DNN
    label:  str    Label for plot title
    """
    sns.set_theme()
    plt.figure(figsize=(10,4))

    # we use a loop to go through key metrics
    for subplot, curve in enumerate(["loss", "accuracy"]):
        plt.subplot(1,2, subplot + 1)
        plt.plot(history.history[curve], label="training")
        plt.plot(history.history["val_"+curve], label="validation")
        plt.legend()
        plt.title(label + ":" + curve)
    
    plt.tight_layout()