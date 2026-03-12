# Heart Disease Classification Pipeline with Synthetic Data Augmentation

## Project Overview
This project aims to develop and evaluate a robust machine learning pipeline for the classification of heart diseases, specifically designed as a decision support tool for medical professionals. A core component of the project is the use of **Generative AI (CTGAN)** to produce synthetic clinical data, which addresses the challenges of small sample sizes and privacy in healthcare datasets.

The research focuses on the consistency of model decision-making across real and synthetic environments using **Explainable AI (XAI)** techniques to ensure clinical transparency and fairness.

## Key Features
* **Synthetic Data Generation:** Utilizing the Synthetic Data Vault (SDV) and CTGAN to augment original clinical datasets.
* **Comparative EDA:** Statistical consistency analysis between real and synthetic data distributions.
* **Comprehensive ML Pipeline:** Includes preprocessing (handling missing data, encoding, scaling), model training, and hyperparameter tuning.
* **Model Benchmarking:** Evaluation of multiple classifiers, including Logistic Regression, Decision Trees, Random Forest, XGBoost, and Deep Neural Networks (DNN).
* **Interpretability Analysis:** Leveraging **SHAP** and feature importance to understand and verify the clinical drivers behind predictions.
* **Interactive Prototype:** A web interface developed with **Gradio** for real-time visualization of predictions and model reasoning.

## Technical Stack
* **Language:** Python 
* **Data Libraries:** NumPy, Pandas 
* **Visualization:** Matplotlib, Seaborn 
* **Machine Learning:** Scikit-Learn, XGBoost, TensorFlow/Keras 
* **Synthetic Data Generation:** SDV (Synthetic Data Vault), CTGAN 
* **Web Interface:** Gradio 

## Repository Structure
To ensure professional standards and maintainability, the project is organized as follows:
* `data/`: Contains original clinical datasets (anonymized).
* `notebooks/`: Experimental stages, including Exploratory Data Analysis (EDA) and playground testing.
* `src/`: Modular source code for data loading, synthesis, and model definitions.
* `models/`: Storage for trained model artifacts and metadata.
* `app/`: Implementation of the Gradio web interface.

## Ethical Considerations
This project utilizes existing, anonymized public data to ensure privacy and compliance with regulations such as GDPR. Synthetic data is employed not only for augmentation but also to enhance privacy and analyze model fairness across diverse subgroups. The tool is strictly intended as a support for expert decisions and not as a definitive diagnostic tool.

## Academic Information
* **University:** University of Hertfordshire 
* **Course:** BSc Computer Science - Applied Data Science 
* **Student:** Riccardo Cracolici 
* **Academic Year:** 2025-2026 
* **Supervisor Project Meeting Progress:** Project refined through structured academic planning and supervisor feedback.