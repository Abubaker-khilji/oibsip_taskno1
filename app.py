import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("gradient_boosting_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page title
st.title("Sales Prediction App")

# Add input fields for the user to enter values
tv = st.number_input("Enter TV Advertising Budget", min_value=0.0, value=100.0)
radio = st.number_input("Enter Radio Advertising Budget", min_value=0.0, value=50.0)
newspaper = st.number_input("Enter Newspaper Advertising Budget", min_value=0.0, value=30.0)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the input data in the correct shape
    features = np.array([tv, radio, newspaper]).reshape(1, -1)
    
    # Predict the sales
    prediction = model.predict(features)
    
    # Display the prediction result
    st.success(f"Predicted Sales: ${prediction[0]:.2f}")

# Add some explanation text
st.markdown("""
This app predicts sales based on the advertising budget for TV, Radio, and Newspaper. 
Enter the advertising budgets and click on "Predict" to see the predicted sales.
""")
