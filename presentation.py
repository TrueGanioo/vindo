import streamlit as st

def presentation_page():
    st.title("Presentation")

    st.header("Slide 1: Introduction")
    st.write("Predictive Maintenance Project")
    st.write("Objective: Predict equipment failures using machine learning.")

    st.header("Slide 2: Dataset Description")
    st.write("Dataset: AI4I 2020 Predictive Maintenance Dataset")
    st.write("Source: UCI Machine Learning Repository")
    st.write("Features: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear")
    st.write("Target: Machine failure (0 or 1)")

    st.header("Slide 3: Workflow")
    st.write("- Data Loading and Preprocessing")
    st.write("- Model Training (Random Forest, XGBoost)")
    st.write("- Model Evaluation (Accuracy, Confusion Matrix, ROC-AUC)")
    st.write("- Streamlit App Development")

    st.header("Slide 4: Results")
    st.write("Random Forest and XGBoost models trained successfully.")
    st.write("Evaluation metrics displayed in the Analysis section.")

    st.header("Slide 5: Conclusion")
    st.write("The project demonstrates a working predictive maintenance solution.")
    st.write("Future improvements: Add more models, improve UI, deploy the app.")
