import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Preprocessing functions
def clean_col(dframe, columns):
    for column in columns:
        if column in dframe.columns:
            dframe[column] = (
                dframe[column]
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(' ', '', regex=False)
                .str.strip()
                .str.lower()
            )
    return dframe

def imputetion(dframe):
    dframe = dframe.replace(['NAN', 'nan', 'NaN', 'None'], np.nan)
    for column in dframe.columns:
        if dframe[column].dtype in ['float64', 'int64']:
            dframe[column] = dframe[column].fillna(dframe[column].mean())
        else:
            mode = dframe[column].mode().iloc[0] if not dframe[column].mode().empty else ''
            dframe[column] = dframe[column].fillna(mode)
    return dframe

def convert_columns_to_categorical(dataframe, columns):
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype('category')
    return dataframe

def ohe(dataframe, columns):
    dataframe_encoded = pd.get_dummies(dataframe, columns=columns, drop_first=True)
    return dataframe_encoded

# Streamlit app UI
st.title("Fake Job Posting Detection")
st.markdown("This app predicts if a job posting is fake or legitimate based on key features.")

# Backend: Fetch dataset and train model
@st.cache_resource
def train_model():
    # Fetch dataset
    dataset_url = r"C:\Users\aniru\OneDrive\Desktop\Sem-I\DIC\Fake_job_posting\Fake_job_posting\fake_job_postings.csv"
    df = pd.read_csv(dataset_url)
    
    # Preprocessing and training pipeline
    col = ['location', 'company_profile', 'department', 'description', 
           'requirements', 'benefits', 'salary_range', 'employment_type', 
           'required_experience', 'required_education', 'industry', 'function']
    df[col] = df[col].fillna('NAN')
    df = df.drop(['salary_range'], axis=1)

    columns = ['title', 'location', 'company_profile', 'description', 
               'requirements', 'benefits', 'employment_type']
    cleaned_df = clean_col(df, columns)
    imputed_df = imputetion(cleaned_df)

    columns_to_convert = ['title', 'location', 'department', 'employment_type', 
                          'company_profile', 'description', 'requirements', 
                          'benefits', 'required_experience', 'required_education', 
                          'industry', 'function']
    df_converted = convert_columns_to_categorical(imputed_df, columns_to_convert)

    col_encode_ohe = ['title', 'location', 'department', 'employment_type', 
                      'company_profile', 'description', 'requirements', 
                      'benefits', 'required_experience', 'required_education', 
                      'industry', 'function']
    df_encoded_ohe = ohe(df_converted, col_encode_ohe)

    # Save the feature names
    feature_columns = df_encoded_ohe.drop(columns=['fraudulent']).columns
    joblib.dump(feature_columns, "feature_columns.pkl")

    numeric_cols = df_encoded_ohe.select_dtypes(include=['float64', 'int64']).columns

    # Split data into features and target
    df_no_outliers = df_encoded_ohe
    X = df_no_outliers.drop(columns=['fraudulent'])
    y = df_no_outliers['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model for prediction
    joblib.dump(model, "fake_job_model.pkl")

# Show spinner while training
with st.spinner("Training the model... Please wait."):
    train_model()
st.success("Model trained successfully! You can now make predictions below.")

# Step 2: User input for prediction
st.header("Make a Prediction")

# Load the saved model and feature columns
try:
    model = joblib.load("fake_job_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except:
    st.error("Error loading model or feature columns. Please ensure training is complete.")

# Input form for prediction
with st.form("prediction_form"):
    title = st.text_input("Job Title")
    location = st.text_input("Job Location")
    company_profile = st.text_input("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    benefits = st.text_input("Job Benefits")
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])

    submit_button = st.form_submit_button("Predict")

if submit_button and model:
    # Convert input into a DataFrame
    input_data = pd.DataFrame({
        "title": [title],
        "location": [location],
        "company_profile": [company_profile],
        "description": [description],
        "requirements": [requirements],
        "benefits": [benefits],
        "employment_type": [employment_type],
    })

    # Preprocess the input data
    input_data = clean_col(input_data, input_data.columns)
    input_data = imputetion(input_data)
    input_data = convert_columns_to_categorical(input_data, input_data.columns)
    input_data = ohe(input_data, input_data.columns)

    # Align input data with feature columns
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("This job posting is predicted to be **FAKE**.")
    else:
        st.success("This job posting is predicted to be **LEGITIMATE**.")
