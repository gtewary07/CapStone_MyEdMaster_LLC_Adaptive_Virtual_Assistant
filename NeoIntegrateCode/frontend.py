# frontend.py

import streamlit as st
import requests

st.title("Question Classification")

question = st.text_input("Enter your question:")

if st.button("Predict"):
    if question.strip():
        response = requests.post("http://127.0.0.1:8000/predict", json={"question": question})
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.success(f"Prediction Vector: {prediction}")
        else:
            st.error("Failed to get a prediction. Please try again.")
    else:
        st.error("Please enter a valid question.")

if st.button("Weak Label"):
    if question.strip():
        response = requests.post("http://127.0.0.1:8000/weak_label", json={"question": question})
        if response.status_code == 200:
            label = response.json().get("prediction")
            st.success(f"Weak Label: {label}")
        else:
            st.error("Failed to get a weak label. Please try again.")
    else:
        st.error("Please enter a valid question.")

data_path = st.text_input("Enter the path to your training data:")
if st.button("Train Model"):
    if data_path.strip():
        response = requests.post("http://127.0.0.1:8000/train", json={"data_path": data_path})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Model trained. Accuracy: {result['accuracy']}")
        else:
            st.error("Failed to train the model. Please check the data path and try again.")
    else:
        st.error("Please enter a valid data path.")