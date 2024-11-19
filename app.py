import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Define the features
selected_features = [
    'concave points_worst', 
    'perimeter_worst', 
    'concave points_mean', 
    'radius_worst', 
    'perimeter_mean', 
    'area_worst', 
    'radius_mean'
]

# Streamlit App
st.title("Breast Cancer Prediction")
st.write("""
This app predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** 
based on selected features from the dataset.
""")

# Sidebar for input
st.sidebar.header("Input Features")

# Initialize a dictionary to store user inputs
user_data = {}
for feature in selected_features:
    user_data[feature] = st.sidebar.number_input(
        f"{feature}:",
        min_value=0.0,  # Set minimum value
        max_value=100000.0,  # Set maximum value
        value=0.0,  # Default value
        step=0.1  # Increment step
    )

# Convert user input to a DataFrame
input_df = pd.DataFrame([user_data])

# Prediction button
if st.button("Predict"):
    # Predict using the loaded model
    prediction = model.predict(input_df)[0]

    # Display result
    result = "Malignant (M)" if prediction == 1 else "Benign (B)"
    st.success(f"The model predicts: **{result}**")
    st.write("Feature Values Provided:")
    st.write(input_df)
