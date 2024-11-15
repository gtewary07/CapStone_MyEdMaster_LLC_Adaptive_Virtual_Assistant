# NeoFront.py
import streamlit as st
import requests

if 'question' not in st.session_state:
    st.session_state.question = ""
if 'system_response' not in st.session_state:
    st.session_state.system_response = ""

st.title("Personalized Question-Answering System")

question = st.text_input("Enter your question:", value=st.session_state.question)
age = st.number_input("Enter your age:", min_value=1, max_value=120)
knowledge_level = st.selectbox(
    "Select your knowledge level:",
    ["beginner", "intermediate", "advanced"]
)

if st.button("Submit"):
    if question.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "question": question,
                    "age": age,
                    "knowledge_level": knowledge_level
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.session_state.system_response = result.get('system_response', 'No response available')
                st.session_state.question = question

                st.info(f"System Response: {st.session_state.system_response}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.RequestException as e:
            st.error(f"Request failed: {str(e)}")
    else:
        st.error("Please enter a valid question.")

user_answer = st.text_area("Please explain the answer in your own words:")

if st.button("Check Understanding"):
    if user_answer.strip():
        try:
            understanding_response = requests.post(
                "http://127.0.0.1:8000/assess_understanding",
                json={
                    "original_question": st.session_state.question,
                    "system_response": st.session_state.system_response,
                    "user_answer": user_answer
                }
            )

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