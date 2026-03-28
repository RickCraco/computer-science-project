import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance(results_list: list):
    """
    Get the feature importance for a classifier
    and plots a bar chart to better visualize 
    the feature importance values.

    Input:
    results_list: list  list containing all trained models
    """
    # we  loop trough each model
    for result in results_list:
        # we retrieve the model name
        model_name = result["model"]

        # we retrieve the best estimator from GridSearchCV for both syn and real
        model_syn = result['best_estimator_syn'].named_steps["classifier"]
        model_real = result['best_estimator_real'].named_steps["classifier"]

        # we retrieve the feature names
        feature_names = model_syn.named_steps['preprocessor'].get_feature_names_out()

        # we check if the model has the right attribute for feature importance
        if hasattr(model_syn, "coef_") and hasattr(model_real, "coef_"):
            importances_syn = model_syn.coef_[0]    # we take the feature importances for both real and syn
            importances_real = model_real.coef_[0]
        elif hasattr(model_syn, "feature_importances_") and hasattr(model_real, "feature_importances_"):
            importances_syn = model_syn.feature_importances_
            importances_real = model_real.feature_importances_
        else:
            print("Model does not support feature importances")
            continue 

        # we save the results inside a df
        df_importance_syn = pd.DataFrame({
            "feature": feature_names,
            "importance_syn": importances_syn,
        }).sort_values(by="importance_syn", ascending=False)

        df_importance_real = pd.DataFrame({
            "feature": feature_names,
            "importance_real": importances_real
        }).sort_values(by="importance_real", ascending=False)

        # we create a subplot for multiple plots (synthetic and real)
        fig, axs = plt.subplots(1,2, figsize=(10,8))  # we create a grid with 1 row and 2 cols

        # bar chart for better visualization
        sns.barplot(data=df_importance_syn, x="importance_syn", y="feature", ax=axs[0])
        axs[0].set_title(f"Feature Importance - {model_name} - Synthetic")

        sns.barplot(data=df_importance_real, x="importance_real", y="feature", ax=axs[1])
        axs[1].set_title(f"Feature Importance - {model_name} - Real")

        plt.tight_layout()
        plt.show()