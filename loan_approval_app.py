import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        with open('loan_approval_model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
            
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model training script has been run successfully.")
        return None, None, None

model, scaler, feature_names = load_model()

# Set page title and description
st.title("Loan Approval Prediction System")
st.write("""
This application predicts the likelihood of loan approval based on applicant information in India.
Fill in the form below and click 'Predict' to see the result.
""")

# Create a sidebar for better organization
st.sidebar.header("Application Form")

# Collect user inputs
with st.sidebar.form("loan_form"):
    # Personal Information
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    # Financial Information
    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=25000)
    coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=9, value=100, 
                                help="Enter loan amount in thousands of rupees (e.g., 100 = ₹100,000)")
    loan_amount_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", ["1.0", "0.0"], 
                                 help="1.0 means credit history meets guidelines, 0.0 means it doesn't")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Function to preprocess the user input
def preprocess_input(user_input, feature_names):
    # Convert to dataframe with a single row
    input_df = pd.DataFrame([user_input])
    
    # One-hot encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Create a dataframe with all feature columns from the training set
    final_df = pd.DataFrame(columns=feature_names)
    
    # Fill in available columns
    for col in input_encoded.columns:
        if col in feature_names:
            final_df[col] = input_encoded[col]
    
    # Fill missing columns with 0 (for one-hot encoded columns that didn't appear in the input)
    final_df = final_df.fillna(0)
    
    return final_df

# Main area for displaying predictions
if submit_button and model is not None:
    try:
        # Prepare user input
        user_input = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': float(loan_amount_term),
            'Credit_History': float(credit_history),
            'Property_Area': property_area
        }
        
        # Preprocess the input
        processed_input = preprocess_input(user_input, feature_names)
        
        # Scale the input
        scaled_input = scaler.transform(processed_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]  # Probability of approval
        
        # Display result with nice formatting
        st.header("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("✅ Loan Approved!")
            else:
                st.error("❌ Loan Not Approved")
        
        with col2:
            st.metric("Approval Probability", f"{probability*100:.1f}%")
        
        # Create a gauge chart for probability
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.barplot(x=[probability, 1-probability], y=["", ""], palette=["green", "lightgrey"], ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title("Approval Likelihood")
        ax.set_xlabel("Probability")
        ax.set_yticks([])
        
        # Add probability marker
        ax.axvline(x=probability, color='red', linestyle='-', linewidth=2)
        
        # Add annotation for the probability value
        ax.text(probability, 0, f"{probability:.2f}", color='black', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check your input values and try again.")

# Add some helpful information at the bottom
st.sidebar.markdown("""
### Tips for Approval
1. **Credit History** is typically the most important factor
2. Higher applicant income improves chances
3. Lower loan amount relative to income improves chances
4. Having a co-applicant can strengthen your application
5. Property in Semiurban areas may have higher approval rates
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Indian Loan Approval Prediction System")