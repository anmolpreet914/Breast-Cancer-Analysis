import streamlit as st
import pandas as pd
import joblib

# Load the dataset and model
df = pd.read_csv('breast_cancer_selected_features.csv')
model = joblib.load('best_ann_model.pkl')

# Streamlit App
st.title('Breast Cancer Prediction App')

st.write("### Breast Cancer Dataset")
st.write(df)

# User input
st.write("### User Input Features")
input_data = {}
for col in df.columns[:-1]:
    input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.write(f"### Prediction: {'Malignant' if prediction[0] == 0 else 'Benign'}")
    st.write(f"### Prediction Probability: {prediction_proba[0]}")