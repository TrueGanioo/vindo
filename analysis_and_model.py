import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

def analysis_and_model_page():
    st.header("Analysis and Model")

    # Data Loading
    st.subheader("Data Loading")
    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        st.success("Dataset loaded successfully!")
        st.write("Data shape:", data.shape)
        st.write("First few rows:", data.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    data = data.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    if data.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Filling with 0.")
        data = data.fillna(0)

    scaler = StandardScaler()
    numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    st.write("Data after preprocessing:", data.head())

    # Data Splitting
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

    # Model Training
    st.subheader("Model Training")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }
    best_model_name = None
    best_accuracy = 0

    if st.button("Train Models"):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            st.write(f"{name} - Accuracy: {accuracy:.2f}, ROC-AUC: {roc_auc:.2f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
        st.success(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.2f})")

    # Model Evaluation
    st.subheader("Model Evaluation")
    if best_model_name:
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f"Confusion Matrix ({best_model_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Predict on New Data
    st.subheader("Predict on New Data")
    with st.form(key='prediction_form'):
        type_options = ['L', 'M', 'H']
        type_input = st.selectbox("Type", type_options)
        air_temp = st.number_input("Air temperature (K)", min_value=290.0, max_value=310.0, value=298.0)
        proc_temp = st.number_input("Process temperature (K)", min_value=300.0, max_value=320.0, value=308.0)
        rot_speed = st.number_input("Rotational speed (rpm)", min_value=1000, max_value=2000, value=1400)
        torque = st.number_input("Torque (Nm)", min_value=30.0, max_value=60.0, value=40.0)
        tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=300, value=0)
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        new_data = pd.DataFrame({
            'Type': [type_input],
            'Air temperature': [air_temp],
            'Process temperature': [proc_temp],
            'Rotational speed': [rot_speed],
            'Torque': [torque],
            'Tool wear': [tool_wear]
        })
        new_data['Type'] = LabelEncoder().fit_transform(new_data['Type'])
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        prediction = best_model.predict(new_data)[0]
        probability = best_model.predict_proba(new_data)[0][1]
        st.write(f"Prediction: {'Failure' if prediction == 1 else 'No Failure'}, Failure Probability: {probability:.2f}")

if __name__ == "__main__":
    analysis_and_model_page()
