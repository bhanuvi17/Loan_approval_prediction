import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

# Set the working directory to where the script is located
# Uncomment and modify this line if needed
# os.chdir('path_to_your_project_directory')

# Load the dataset
# Update the path to where your dataset is actually located
df = pd.read_csv("loan_prediction.csv")
# If you're having path issues, use the absolute path like:
# df = pd.read_csv(r"C:\Users\bhanu\OneDrive\Desktop\project\Loan_approval_prediction\loan_prediction.csv")

print("Dataset loaded successfully with", df.shape[0], "rows and", df.shape[1], "columns")
print("This is an Indian loan dataset with values in Indian Rupees (₹)")

# Preprocessing
df.drop('Loan_ID', axis=1, inplace=True)

# Handling missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']:
    df[col].fillna(df[col].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
print("Missing values handled successfully")

# Remove outliers from ApplicantIncome
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

df = remove_outliers(df, 'ApplicantIncome')
df = remove_outliers(df, 'CoapplicantIncome')
print(f"Outliers removed, new dataset size: {df.shape[0]} rows")

# Convert categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 
                                         'Self_Employed', 'Property_Area'])

print("One-hot encoding completed. Total features:", df_encoded.shape[1])

# Define features and target
X = df_encoded.drop('Loan_Status', axis=1)
y = df_encoded['Loan_Status'].map({'Y': 1, 'N': 0})

print("Class distribution before SMOTE:")
print(y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_sm).value_counts())

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

print("Data scaling completed")

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=None,       # Maximum depth of trees
    min_samples_split=2,  # Minimum samples required to split
    min_samples_leaf=1,   # Minimum samples at leaf node
    random_state=42
)

print("Training Random Forest model...")
rf_model.fit(X_train_scaled, y_train_sm)
print("Model training completed")

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Save the model and scaler
with open('loan_approval_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
    
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)

print("Model and preprocessing components saved successfully!")
print("Files saved: loan_approval_model.pkl, scaler.pkl, feature_names.pkl")
print("These files should be in the same directory as your Streamlit app")