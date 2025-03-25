import os
from groq import Groq
import streamlit as st
import time
import json
import hashlib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

client = Groq(
    api_key="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"
)

############################################
# Utility Functions for Evaluation & Feedback
############################################

def check_response_similarity(system_response, user_response):
    """Check if user response is copied from system response"""
    # IMPROVEMENT IDEA: Add more sophisticated similarity checks:
    # - Semantic similarity using embeddings
    # - Paraphrase detection
    # - Add threshold as config parameter
    system_clean = " ".join(system_response.lower().split())
    user_clean = " ".join(user_response.lower().split())
    if system_clean == user_clean:
        return True
    system_words = set(system_clean.split())
    user_words = set(user_clean.split())
    similarity = len(system_words.intersection(user_words)) / len(system_words)
    return similarity > 0.8  # IMPROVEMENT: Make threshold configurable


def generate_improvement_feedback(category, score):
    """Generate specific improvement suggestions based on category and score"""
    feedback = {
        "0": """
🚫 Plagiarism Detected:
- Your response appears to be copied from the system answer.
- Please rewrite the answer in your own words.
- Demonstrate your understanding through original explanation.
- Try to connect concepts with your own experiences.
""",
    }
    return feedback.get(category, "Please provide a valid response to receive improvement feedback.")


def calculate_score(evaluation):
    """Calculate score based on the evaluation response."""
    # IMPROVEMENT IDEA: Add more robust score calculation:
    # - Weighted scoring for different aspects
    # - Normalization across different evaluation types
    # - Confidence scoring
    try:
        if isinstance(evaluation, dict):
            score = evaluation.get("Score", 0)
            return round(score, 3)
        elif isinstance(evaluation, str):
            return 0.000
        else:
            return 0.000
    except Exception as e:
        # IMPROVEMENT: Add error logging
        return 0.000


def determine_topic_complexity(content):
    """
    Analyze the content to determine its complexity and suggest an appropriate number of questions.
    Returns a complexity level (1-5) and recommended number of questions.
    """
    # IMPROVEMENT IDEA: Add caching for complexity analysis
    # IMPROVEMENT IDEA: Make prompt more configurable
    prompt = (
        f"Analyze the following educational content and determine its complexity on a scale of 1-5, "
        f"where 1 is very simple and 5 is very complex. Based on the complexity, suggest an appropriate "
        f"number of questions to test understanding (2-3 questions for simple topics, 4-6 for moderately "
        f"complex topics, and 7-10 for very complex topics). Also, suggest a distribution of question types "
        f"(rationale, factual, procedural) that would be appropriate for this content. "
        f"Return your response in JSON format with keys 'complexity_level', 'recommended_questions', and "
        f"'question_distribution' (an object with keys 'rationale', 'factual', 'procedural' and values as percentages). "
        f"Content: {content[:4000]}"  # IMPROVEMENT: Make content length limit configurable
    )

    response, _ = get_groq_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
        st.session_state.user_data['experience'],
        conversation_history=None
    )

    try:
        start_index = response.find('{')
        end_index = response.rfind('}')
        if start_index != -1 and end_index != -1:
            json_string = response[start_index:end_index + 1]
            analysis = json.loads(json_string)
        else:
            analysis = json.loads(response)

        return {
            'complexity_level': analysis.get('complexity_level', 3),
            'recommended_questions': analysis.get('recommended_questions', 5),
            'question_distribution': analysis.get('question_distribution',
                                                {'rationale': 33, 'factual': 33, 'procedural': 34})
        }
    except Exception as e:
        # IMPROVEMENT IDEA: Add error logging and notification
        st.error(f"Error analyzing topic complexity: {str(e)}")
        return {
            'complexity_level': 3,
            'recommended_questions': 5,
            'question_distribution': {'rationale': 33, 'factual': 33, 'procedural': 34}
        }


############################################
# Session State & User Data Functions
############################################

def init_session_state():
    """Initialize session state variables"""
    # IMPROVEMENT IDEA: Add session expiration timer
    # IMPROVEMENT IDEA: Add data validation for session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False


def save_user_data(username, password, age, interest, experience):
    """Save user data to a JSON file"""
    try:
        users = {}
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users = json.load(f)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        users[username] = {
            'password': hashed_password,
            'age': age,
            'interest': interest,
            'experience': experience,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open('users.json', 'w') as f:
            json.dump(users, f)
        return True
    except Exception as e:
        # IMPROVEMENT IDEA: Add error logging
        st.error(f"Error saving user data: {str(e)}")
        return False

def main_application():
    """Main application interface"""

    st.title(f"Welcome {st.session_state.user_data.get('interest', 'Enthusiast')}!")

def main():
    """Main program execution"""

    init_session_state()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_application()
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()
