import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import shap
import joblib

# we load the pipeline
best_syn = joblib.load("../models/catboost_syn.pkl")

# we initialize the SHAP explainer
explainer = shap.Explainer(best_syn)

# feature order expected by the pipeline
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

# main function
def predict_result(df: pd.DataFrame) -> tuple:
    """
    Predict the class label with the associated probability
    and plot the Waterfall plot for SHAP analysis.

    Input:
    df:  pd.DataFrame  single row of data input by the user

    Output:

    tuple:  tuple  tuple containing the class label, probability and figure of the plot
    """
    # input dataframe
    input_data = pd.DataFrame(df, columns=FEATURES)

    # we make the model prediction
    pred = best_syn.predict(input_data)[0]
    proba = best_syn.predict_proba(input_data)[0][1]

    # we calculate the SHAP values
    shap_values = explainer(input_data)

    # we plot the waterfall plot
    fig = plt.figure()
    shap.waterfall_plot(shap_values[0], max_display=20)

    # we calculate the class label "Heart Disease"
    label = "Heart Disease" if pred == 1 else "No Heart Disease"

    return label, float(proba), fig


# gradio user interface
demo = gr.Interface(
    fn = predict_result,
    inputs=gr.DataFrame(
        headers=FEATURES,
        row_count=1,
        col_count=len(FEATURES),
        datatype=[
            "number", "str", "str", "number", "number",
            "number", "str", "number", "str", "number", "str"
        ],
        label="User Data Input"
    ),
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