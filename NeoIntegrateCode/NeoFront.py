import streamlit as st
import requests

# Initialize session state variables
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'system_response' not in st.session_state:
    st.session_state.system_response = ""
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'similarity_score' not in st.session_state:
    st.session_state.similarity_score = None

st.title("Question Classification and Response System")

# Use session state for the question input
question = st.text_input("Enter your question:", value=st.session_state.question)

if st.button("Submit"):
    if question.strip():
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json={"question": question})

            if response.status_code == 200:
                result = response.json()
                st.session_state.prediction = result.get('prediction', 'N/A')
                st.session_state.system_response = result.get('system_response', 'No response available')
                st.session_state.similarity_score = result.get('similarity_score', 'N/A')
                st.session_state.question = question
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.RequestException as e:
            st.error(f"Request failed: {str(e)}")
    else:
        st.error("Please enter a valid question.")

# Display results if available
if st.session_state.prediction is not None:
    st.success(f"Prediction Vector: {st.session_state.prediction}")
    st.info(f"System Response: {st.session_state.system_response}")
    st.info(f"Similarity Score: {st.session_state.similarity_score}")

    # Add a text area for the user to provide their understanding
    user_answer = st.text_area("Please explain the answer in your own words:")

    if st.button("Check Understanding"):
        if user_answer.strip():
            try:
                understanding_response = requests.post("http://127.0.0.1:8000/assess_understanding",
                                                       json={"original_question": st.session_state.question,
                                                             "system_response": st.session_state.system_response,
                                                             "user_answer": user_answer})

                if understanding_response.status_code == 200:
                    understanding_result = understanding_response.json()
                    understanding_score = understanding_result.get('understanding_score', 'N/A')
                    st.success(f"Your understanding score: {understanding_score}%")

                    if understanding_score > 80:
                        st.balloons()
                        st.success("Great job! You have a good understanding of the answer.")
                    elif understanding_score > 50:
                        st.info("You have a decent understanding, but there's room for improvement.")
                    else:
                        st.warning("It seems you might need to review the answer. Try rephrasing it in your own words.")
                else:
                    st.error(f"Error: {understanding_response.status_code} - {understanding_response.text}")
            except requests.RequestException as e:
                st.error(f"Request failed: {str(e)}")
        else:
            st.error("Please provide your explanation before checking understanding.")