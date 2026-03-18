import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.preprocessing import get_preprocessor

# for DNN
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout


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



def evaluate_synthetic_quality(model, param_grid: dict, syn_df: pd.DataFrame, train_real_df: pd.DataFrame, test_real_df: pd.DataFrame, num_cols: list, cat_cols: list, target_col='HeartDisease'):
    """
    Train and evaluate a model using synthetic data
    and a model using real data using the Train Synthetic
    Test Real (TSTR) standard, to evaluate the synthetic data
    efficacy.

    Input:
    model: obj   Model to train and evaluate
    param_grid: dict  Dictionary for hyperparameter tuning
    syn_df: pd.DataFrame  Synthetic dataset
    train_real_df: pd.DataFrame  Real dataset used for training (70%)
    test_real_df: pd.DataFrame   Real dataset used for testing (30%)
    num_cols: list  Numerical features
    cat_cols: list  Categorical features

    Output:
    results: dict  Dictionary containing best estimator, score and efficacy
    """
    # preparing the synthetic and real data
    X_syn = syn_df.drop(target_col, axis=1)
    y_syn = syn_df[target_col]

    X_train_real = train_real_df.drop(target_col, axis=1)
    y_train_real = train_real_df[target_col]

    X_test_real = test_real_df.drop(target_col, axis=1)
    y_test_real = test_real_df[target_col]

    # building the complete pipeline
    preprocessor = get_preprocessor(num_cols, cat_cols)  # we use our preprocessor

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # training on synthetic data and testing on real (TSTR)
    print(f"\n Training {model.__class__.__name__} model on synthetic data")
    grid_syn = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=5,  # cross-validation folds
        scoring='accuracy',
        n_jobs=-1
    )
    grid_syn.fit(X_syn, y_syn)  # we train the model on synthetic data
    acc_syn_on_real = grid_syn.best_estimator_.score(X_test_real, y_test_real)

    # training on real data and testing on real data our benchmark
    print(f"\n Training {model.__class__.__name__} model on real data")
    grid_real = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_real.fit(X_train_real, y_train_real)
    acc_real_on_real = grid_real.best_estimator_.score(X_test_real, y_test_real)

    # reporting results
    print("\n" + "="*40)
    print(f"Model: {model.__class__.__name__}")
    print(f"Best params SYNTHETIC: {grid_syn.best_params_}")
    print(f"Accuracy TRTR (Real on Real): {acc_real_on_real:.4f}")
    print(f"Accuracy TSTR (Synthetic on Real): {acc_syn_on_real:.4f}")

    # we calculate the efficacy of the synthetic data
    efficacy = (acc_syn_on_real / acc_real_on_real) * 100 if acc_real_on_real > 0 else 0
    print(f"Sythetic Data Efficacy: {efficacy:.2f}%")
    print("="*40)

    results = {
        'model': model.__class__.__name__,
        'acc_real': acc_real_on_real,
        'acc_syn': acc_syn_on_real,
        'efficacy': efficacy,
        'best_estimator_syn': grid_syn.best_estimator_,
        'best_estimator_real': grid_real.best_estimator_
    }

    return results