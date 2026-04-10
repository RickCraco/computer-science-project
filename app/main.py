import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import shap
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "catboost_syn.pkl")

# we load the pipeline
best_syn = joblib.load(model_path)
preprocessor = best_syn.named_steps["preprocessor"]
classifier = best_syn.named_steps["classifier"]
feature_names = preprocessor.get_feature_names_out()

# feature order expected by the pipeline
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

# main function
def predict_result(*args) -> tuple:
    """
    Predict the class label with the associated probability
    and plot the Waterfall plot for SHAP analysis.

    Input:
    df:  pd.DataFrame  single row of data input by the user

    Output:

    tuple:  tuple  tuple containing the class label, probability and figure of the plot
    """
    data = dict(zip(FEATURES, args))

    # input dataframe
    input_data = pd.DataFrame([data])

    # we make the model prediction
    pred = best_syn.predict(input_data)[0]
    proba = best_syn.predict_proba(input_data)[0][1]

    # we initialize the SHAP explainer and calculate SHAP values
    input_transformed = preprocessor.transform(input_data) # SHAP does not work well with pipelines, so we need to pass the preprocessed input

    explainer = shap.Explainer(classifier,feature_names=feature_names)
    shap_values = explainer(input_transformed)

    # SHAP often draws on the current matplotlib figure and may return None.
    # for Gradio's Plot output, we must return an explicit Figure object.
    plt.close("all")
    shap.waterfall_plot(shap_values[0], max_display=20, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    # we calculate the class label "Heart Disease"
    label = "Heart Disease" if pred == 1 else "No Heart Disease"

    return label, float(proba), fig


# gradio user interface
demo = gr.Interface(
    fn = predict_result,
    inputs= [
        gr.Number(label="Age"),
        gr.Dropdown(["M", "F"], label="Sex"),
        gr.Dropdown(["TA", "ATA", "NAP", "ASY"], label="Chest Pain Type"),
        gr.Number(label="Resting Blood Pressure"),
        gr.Number(label="Cholesterol"),
        gr.Dropdown([0, 1], label="Fasting Blood Sugar"),
        gr.Dropdown(["Normal", "ST", "LVH"], label="Resting ECG"),
        gr.Number(label="Max Heart Rate"),
        gr.Dropdown(["Y", "N"], label="Exercise Induced Angina"),
        gr.Number(label="Oldpeak"),
        gr.Dropdown(["Up", "Flat", "Down"], label="ST Slope"),
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Probability (Heart Disease)"),
        gr.Plot(label="SHAP Waterfall Plot")
    ],
    title="Heart Disease Prediction App",
    description="""Enter a user’s details to receive a prediction and obtain an explanation via SHAP.

                DISCLAIMER: This app is not intended to serve as a definitive substitute for a doctor, but is designed to be a tool to support decision-making by medical professionals. Any use of this tool in a real-world setting should not be taken lightly; instead, every aspect and limitation of the system must be taken into account."""
)

if __name__ == "__main__":
    demo.launch(share=True)