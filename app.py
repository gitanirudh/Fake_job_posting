import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature columns
model = joblib.load("forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Preprocessing functions
def clean_col(dframe, columns):
    for column in columns:
        if column in dframe.columns:
            dframe[column] = (
                dframe[column]
                .astype(str)
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(' ', '', regex=False)
                .str.strip()
                .str.lower()
            )
    return dframe

def imputation(dframe):
    dframe = dframe.replace(['NAN', 'nan', 'NaN', 'None'], np.nan)
    for column in dframe.columns:
        mode = dframe[column].mode().iloc[0] if not dframe[column].mode().empty else ''
        dframe[column] = dframe[column].fillna(mode)
    return dframe

# Streamlit UI
st.title("🕵️‍♂️ Fake Job Posting Detection")
st.markdown("Predict if a job posting is **Fake** or **Legitimate** based on key features.")

st.header("📋 Enter Job Details")

with st.form("prediction_form"):
    title = st.text_input("Job Title")
    location = st.text_input("Job Location")
    company_profile = st.text_input("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    benefits = st.text_input("Job Benefits")
    employment_type = st.selectbox("Employment Type", 
                                   ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    if not title or not location or not description:
        st.error("Please fill out at least Title, Location, and Description.")
    else:
        # Build DataFrame
        input_data = pd.DataFrame({
            "title": [title],
            "location": [location],
            "company_profile": [company_profile],
            "description": [description],
            "requirements": [requirements],
            "benefits": [benefits],
            "employment_type": [employment_type],
        })

        # Preprocess input
        input_data = clean_col(input_data, input_data.columns)
        input_data = imputation(input_data)

        # One-hot encode user input
        input_encoded = pd.get_dummies(input_data)

        # Align columns with training feature set
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)
        proba = model.predict_proba(input_encoded)[0][1] * 100

        if prediction[0] == 1:
            st.error(f"🚨 Predicted as **FAKE** job posting.\n\n**Confidence:** {proba:.2f}%")
        else:
            st.success(f"✅ Predicted as **LEGITIMATE** job posting.\n\n**Confidence:** {proba:.2f}%")
