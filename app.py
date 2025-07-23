import streamlit as st
import pandas as pd
import joblib # To load pre-trained scikit-learn objects like LabelEncoder and the model itself

# --- Load Preprocessing Artifacts ---
# Ensure these files are in the same directory as your app.py
try:
    le = joblib.load('label_encoder_gender.joblib')
    x_train_columns = joblib.load('x_train_columns.joblib')
    education_mapping = joblib.load('education_mapping.joblib')
    job_title_edited = joblib.load('job_title_edited.joblib')
    # If you have a trained model, load it here as well
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Error: Preprocessing artifacts not found. Make sure 'label_encoder_gender.joblib', 'x_train_columns.joblib', 'education_mapping.joblib', and 'job_title_edited.joblib' are in the same directory as this script.")
    st.stop() # Stop the app if essential files are missing

# --- Preprocessing Function (from your original code) ---
def preprocess_input_data(input_data):
    """
    Preprocesses a single input data point for salary prediction.

    Args:
        input_data (dict): A dictionary containing the input features
                            (Age, Gender, Education Level, Job Title,
                            Years of Experience).

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed input data,
                          ready for prediction.
    """
    # Create a pandas Series from the input dictionary for easier processing
    input_series = pd.Series(input_data)

    # Map Education Level
    input_series['Education Level'] = education_mapping.get(input_series['Education Level'], -1)

    # Label encode Gender
    try:
        input_series['Gender'] = le.transform([input_series['Gender']])[0]
    except ValueError:
        st.warning(f"Unseen gender category '{input_series['Gender']}' encountered. Please ensure gender is 'Male' or 'Female'.")
        # You might want to set a default value or return an error state here
        return pd.DataFrame() # Return empty DataFrame if an issue occurs

    # Handle Job Title: Check if it's one of the less frequent titles
    job_title_input = input_series['Job Title'] # Store original job title
    if job_title_input in job_title_edited:
        input_series['Job Title'] = 'Others'

    # Create a DataFrame for the single input row to apply one-hot encoding
    # We create it here before dropping 'Job Title' later
    temp_df = pd.DataFrame([input_series])

    # One-hot encode Job Title
    job_title_dummies = pd.get_dummies(temp_df['Job Title'], prefix='Job Title')

    # Align columns with x_train_columns for job title dummies
    # Extract only the 'Job Title_' columns from x_train_columns for reindexing
    job_title_cols_from_xtrain = [col for col in x_train_columns if col.startswith('Job Title_')]
    job_title_dummies = job_title_dummies.reindex(columns=job_title_cols_from_xtrain, fill_value=0)

    # Drop the original 'Job Title' column from the temporary DataFrame
    # Now temp_df contains 'Age', 'Gender', 'Education Level', 'Years of Experience'
    temp_df = temp_df.drop('Job Title', axis=1)

    # Now, combine the numerical/encoded features with the one-hot encoded job titles
    # We only need the columns that are present in temp_df AND are in x_train_columns
    # This prevents the KeyError by ensuring we only select columns that exist in temp_df
    # and are not job title dummies (as they are handled separately).
    non_job_title_and_existing_cols = [col for col in x_train_columns if not col.startswith('Job Title_')]
    
    # Ensure all required numerical/categorical columns are present in temp_df
    # Add any missing columns from non_job_title_and_existing_cols to temp_df with default 0 if necessary
    for col in non_job_title_and_existing_cols:
        if col not in temp_df.columns:
            temp_df[col] = 0 # Or a suitable default value based on your data

    # Select and reorder the non-job-title columns from temp_df
    # Ensure temp_df only contains the relevant non-job-title columns
    preprocessed_data = temp_df[non_job_title_and_existing_cols]

    # Concatenate the preprocessed numerical/encoded features and the new dummy columns
    preprocessed_data = pd.concat([preprocessed_data, job_title_dummies], axis=1)

    # Finally, ensure the entire DataFrame has the exact same columns and order as x_train_columns
    preprocessed_data = preprocessed_data.reindex(columns=x_train_columns, fill_value=0)

    return preprocessed_data

# --- Streamlit UI (rest of your app remains the same) ---
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💰 Salary Prediction App")

st.markdown("""
Enter the details below to get a predicted salary!
---
""")

# Input fields
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])

# Ensure this list is comprehensive based on your training data's unique job titles
# You can generate this list dynamically during training and save it as a joblib file too.
job_title_options = [
    'Software Engineer', 'Data Scientist', 'Project Manager', 'Sales Representative',
    'Marketing Analyst', 'HR Manager', 'Financial Analyst', 'Accountant',
    'Graphic Designer', 'Operations Manager', 'Sales Manager',
    'Content Marketing Manager', 'Data Analyst', 'Digital Marketing Manager',
    'Director of Data Science', 'Director of HR', 'Director of Marketing',
    'Front End Developer', 'Front end Developer', 'Full Stack Engineer',
    'Human Resources Coordinator', 'Human Resources Manager', 'Junior HR Coordinator',
    'Junior HR Generalist', 'Junior Marketing Manager', 'Junior Sales Associate',
    'Junior Sales Representative', 'Junior Software Developer', 'Junior Software Engineer',
    'Junior Web Developer', 'Marketing Coordinator', 'Marketing Director',
    'Product Designer', 'Product Manager', 'Receptionist', 'Research Director',
    'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Executive',
    'Senior Data Scientist', 'Senior HR Generalist', 'Senior Human Resources Manager',
    'Senior Product Marketing Manager', 'Senior Project Engineer',
    'Senior Research Scientist', 'Software Developer', 'Software Engineer Manager',
    'Web Developer', 'Others'
]
job_title = st.selectbox("Job Title", sorted(job_title_options)) # Sort for better UX
years_of_experience = st.slider("Years of Experience", 0.0, 30.0, 5.0, 0.5)

# Create input dictionary
input_data = {
    'Age': float(age),
    'Gender': gender,
    'Education Level': education_level,
    'Job Title': job_title,
    'Years of Experience': float(years_of_experience)
}

if st.button("Predict Salary"):
    with st.spinner("Preprocessing data and making prediction..."):
        preprocessed_df = preprocess_input_data(input_data)

        if not preprocessed_df.empty:
            st.subheader("Preprocessed Data (for inspection):")
            st.dataframe(preprocessed_df)

            
            if 'model' in locals(): # Check if the model was loaded successfully
                try:
                    prediction = model.predict(preprocessed_df)[0]
                    st.success(f"**Predicted Salary:** {prediction:,.2f}INR")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.info("Model not loaded. Cannot make a prediction. Please ensure 'your_trained_model.joblib' is available.")
            # st.info("Prediction functionality is commented out. Uncomment and load your model to enable it.")
        else:
            st.error("Failed to preprocess data. Please check inputs and ensure all necessary files are present.")

