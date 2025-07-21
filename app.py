import streamlit as st 
import numpy as np
import sklearn
import joblib
model = joblib.load(open('C:/Users/karan/OneDrive/Documents/Employee slary prediction/model.pkl','rb'))

# Create the web app title
st.title('Employee Salary Prediction')

# Create input fields
experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, step=0.5)
gender = st.selectbox('Gender', options=['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, step=1)

# Convert gender to numeric
gender_numeric = 1 if gender == 'Male' else 0

# Create a predict button
if st.button('Predict Salary'):
    # Create input array for prediction
    input_data = [[experience,age, gender_numeric]]
    
    # Make prediction (model returns log-transformed salary)
    log_prediction = model.predict(input_data)
    
    # Convert log prediction back to actual salary
    prediction = np.expm1(log_prediction)
    
    # Display the prediction
    st.success(f'Predicted Salary: {prediction[0]:,.2f}')

