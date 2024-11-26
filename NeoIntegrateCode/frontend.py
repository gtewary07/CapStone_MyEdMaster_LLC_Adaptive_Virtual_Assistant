# Import necessary libraries
import streamlit as st  # Streamlit for creating the web interface
import requests         # Requests for making HTTP requests to the backend

# Set the title of the Streamlit application
st.title("Question Classification")

# Input field for the user to enter a question
question = st.text_input("Enter your question:")

# "Predict" button logic
if st.button("Predict"):
    # Ensure that the input question is not empty or just whitespace
    if question.strip():
        # Send a POST request to the backend API to get a prediction
        response = requests.post("http://127.0.0.1:8000/predict", json={"question": question})
        
        # Check if the response from the server is successful
        if response.status_code == 200:
            # Extract and display the prediction result from the API response
            prediction = response.json().get("prediction")
            st.success(f"Prediction Vector: {prediction}")
        else:
            # Display an error message if the API call fails
            st.error("Failed to get a prediction. Please try again.")
    else:
        # Display an error message if the input question is invalid
        st.error("Please enter a valid question.")

# "Weak Label" button logic
if st.button("Weak Label"):
    # Ensure that the input question is not empty or just whitespace
    if question.strip():
        # Send a POST request to the backend API to get a weak label
        response = requests.post("http://127.0.0.1:8000/weak_label", json={"question": question})
        
        # Check if the response from the server is successful
        if response.status_code == 200:
            # Extract and display the weak label from the API response
            label = response.json().get("prediction")
            st.success(f"Weak Label: {label}")
        else:
            # Display an error message if the API call fails
            st.error("Failed to get a weak label. Please try again.")
    else:
        # Display an error message if the input question is invalid
        st.error("Please enter a valid question.")

# Input field for the user to specify the path to their training data
data_path = st.text_input("Enter the path to your training data:")

# "Train Model" button logic
if st.button("Train Model"):
    # Ensure that the data path is not empty or just whitespace
    if data_path.strip():
        # Send a POST request to the backend API to train the model using the provided data path
        response = requests.post("http://127.0.0.1:8000/train", json={"data_path": data_path})
        
        # Check if the response from the server is successful
        if response.status_code == 200:
            # Extract and display the training result (e.g., accuracy) from the API response
            result = response.json()
            st.success(f"Model trained. Accuracy: {result['accuracy']}")
        else:
            # Display an error message if the API call fails
            st.error("Failed to train the model. Please check the data path and try again.")
    else:
        # Display an error message if the data path is invalid
        st.error("Please enter a valid data path.")
