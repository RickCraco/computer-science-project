import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance(model, preprocessor):
    """
    Get the feature importance for a classifier
    and plots a bar chart to better visualize 
    the feature importance values.

    Input:
    model:  obj  Classifier from GridSearchCV pipeline
    preprocessor:   obj  preprocessor from GridSearchCV pipeline
    """
    model_name = type(model.named_steps["classifier"]).__name__
    clf = model.named_steps["classifier"]   # we retrieve the model classifier
    feature_names = preprocessor.get_features_names_out() # we retrieve the feature names from the preprocessor 

    # we check if the classifier has the coef_ or feature_importances attribute
    if hasattr(clf, "coef_"):
        importances = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        print("Model does not support feature importance")

    # we save these values in a dataframe
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    # we plot the barplot for better visualization
    sns.barplot(data=feat_imp_df, x="importance", y="feature")
    plt.title(f"Feature Importance - {model_name}")
    plt.show()