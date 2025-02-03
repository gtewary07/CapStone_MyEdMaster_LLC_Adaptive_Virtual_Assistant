import os
from groq import Groq
import streamlit as st
import time
import json
import hashlib
import pandas as pd
from datetime import datetime

# Initialize Groq client
client = Groq(
    api_key="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"
)

# Detects plagiarism by comparing the user's response with the system-generated answer
def check_response_similarity(system_response, user_response):
    """Check if user response is copied from system response"""
    # Clean responses for comparison
    system_clean = " ".join(system_response.lower().split())
    user_clean = " ".join(user_response.lower().split())

    # Check for exact match or substantial copying
    if system_clean == user_clean:
        return True

    # Check for partial copying (if more than 80% similar)
    system_words = set(system_clean.split())
    user_words = set(user_clean.split())
    similarity = len(system_words.intersection(user_words)) / len(system_words)

    return similarity > 0.8

# Dynamically provides feedback based on the response category and score
def generate_improvement_feedback(category, score):
    """Generate specific improvement suggestions based on category and score"""
    feedback = {
        "0": """
🚫 Plagiarism Detected:
- Your response appears to be copied from the system answer
- Please rewrite the answer in your own words
- Demonstrate your understanding through original explanation
- Try to connect concepts with your own experiences""",

        "1": f"""
✅ Strong Understanding (Score: {score:.3f}/100):
- Good grasp of core concepts
To improve further:
- Add more specific examples from your experience
- Explain practical applications
- Connect concepts to real-world scenarios""",

        "2": f"""
📝 Basic Understanding (Score: {score:.3f}/100):
To improve your response:
- Include missing key concepts
- Provide more detailed technical explanations
- Add examples from your professional experience
- Focus on system components and their relationships""",

        "3": f"""
🌟 Advanced Understanding (Score: {score:.3f}/100):
Excellent technical knowledge! To perfect your response:
- Structure your answer more clearly
- Prioritize most relevant information
- Consider adding industry-specific examples
- Connect concepts to your professional experience""",

        "4": f"""
⚠️ Needs Improvement (Score: {score:.3f}/100):
To correct your response:
- Review the fundamental concepts
- Focus on accuracy of technical details
- Avoid assumptions
- Start with basic definitions
- Connect ideas logically"""
    }

    return feedback.get(category, "Please provide a valid response to receive improvement feedback.")

# Extracts and rounds off the evaluation score from the response dictionary.
def calculate_score(evaluation):
    """Calculate score based on the evaluation response."""
    try:
        if isinstance(evaluation, dict):
            score = evaluation.get("Score", 0)
            return round(score, 3)
        elif isinstance(evaluation, str):
            return 0.000
        else:
            return 0.000
    except Exception as e:
        return 0.000

# Evaluates the user's response, checking for plagiarism and assigning a category with an improvement suggestion.
def evaluate_response(system_answer, user_response):
    """Enhanced evaluation with plagiarism check and improvement feedback"""
    # First check for copying
    if check_response_similarity(system_answer, user_response):
        return {
            "Category": "0",
            "Score": 0.000,
            "Explanation": "🚫 Response appears to be copied from the system answer.",
            "Improvement": generate_improvement_feedback("0", 0.000)
        }

    try:
        # Analyze response content and determine category
        if any(term in user_response.lower() for term in [
            'fpga', 'advanced', 'protocols', 'standards',
            'cross-compilation', 'machine learning'
        ]):
            score = round(90.000 + (len(user_response.split()) / 100), 3)
            category = "3"
            explanation = "🌟 Demonstrates advanced knowledge beyond core concepts"

        elif all(concept in user_response.lower() for concept in [
            'hardware', 'software', 'specific', 'components',
            'microcontroller', 'memory'
        ]):
            score = round(85.000 + (len(user_response.split()) / 100), 3)
            category = "1"
            explanation = "✅ Fully captures main concepts with clear understanding"

        elif any(basic in user_response.lower() for basic in [
            'system', 'computer', 'control', 'device'
        ]):
            score = round(60.000 + (len(user_response.split()) / 100), 3)
            category = "2"
            explanation = "📝 Shows basic understanding but lacks key details"

        else:
            score = round(25.000 + (len(user_response.split()) / 100), 3)
            category = "4"
            explanation = "⚠️ Contains technical inaccuracies or misconceptions"

        return {
            "Category": category,
            "Score": score,
            "Explanation": explanation,
            "Improvement": generate_improvement_feedback(category, score)
        }

    except Exception as e:
        return f"Evaluation error: {str(e)}"
        
# Sets up session variables in Streamlit for tracking user data, scores, and evaluations.
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'scores' not in st.session_state:
        st.session_state.scores = []
    if 'system_response' not in st.session_state:
        st.session_state.system_response = ""
    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []

# Stores user registration details securely (password is hashed before saving)
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
        st.error(f"Error saving user data: {str(e)}")
        return False

# Checks user login credentials by matching hashed passwords from the stored data
def verify_user(username, password):
    """Verify user credentials"""
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                users = json.load(f)
                if username in users:
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    if users[username]['password'] == hashed_password:
                        return users[username]
        return None
    except Exception as e:
        st.error(f"Error verifying user: {str(e)}")
        return None

# Generates a personalized AI response based on user details, retrying up to 3 times if an error occurs
def get_groq_response(prompt, age, interest, experience, retries=3, delay=2):
    """Enhanced prompt generation with user profile"""
    system_prompt = f"""You are a tutor for a {age} year old person interested in {interest} with {experience} experience.
    Adjust your explanation accordingly and provide clear, age-appropriate information that aligns with their professional background."""

    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile"
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return f"An error occurred: {str(e)}"

# Handles user authentication, including login and registration via Streamlit forms
def login_page():
    """Display login page"""
    st.title("Interactive Learning System")
    st.header("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user_data = verify_user(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_data = user_data
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            age = st.number_input("Age", min_value=5, max_value=100, value=25)
            interest = st.text_input("Area of Interest/Hobby")
            experience = st.text_input("Professional Experience")
            submitted = st.form_submit_button("Register")

            if submitted:
                if save_user_data(new_username, new_password, age, interest, experience):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed")


def main_application():
    """Main application interface"""
    st.title(f"Welcome {st.session_state.user_data.get('interest', '')} Enthusiast!")

    # User Profile Display in Sidebar
    st.sidebar.header("Your Profile")
    st.sidebar.write(f"Interest: {st.session_state.user_data.get('interest', '')}")
    st.sidebar.write(f"Experience: {st.session_state.user_data.get('experience', '')}")

    tab1, tab2 = st.tabs(["Q&A Mode", "Learning Assessment Mode"])

    with tab1:
        st.header("Ask Questions")
        with st.form(key='qa_form'):
            question = st.text_input("Enter your question:")
            submit_qa = st.form_submit_button("Get Answer")
            if submit_qa and question:
                response = get_groq_response(
                    question,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['interest'],
                    st.session_state.user_data['experience']
                )
                st.write("Answer:", response)

    with tab2:
        st.header("Learning Assessment")

        # Display previous scores if available
        if st.session_state.scores:
            st.subheader("Your Learning Progress")
            df = pd.DataFrame(st.session_state.scores)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.line_chart(df.set_index('timestamp')['score'])

        # Learning Question Form
        with st.form(key='learning_form'):
            learning_question = st.text_input("What would you like to learn about?")
            submit_learning = st.form_submit_button("Start Learning")

            if submit_learning and learning_question:
                st.session_state.system_response = get_groq_response(
                    learning_question,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['interest'],
                    st.session_state.user_data['experience']
                )
                st.markdown("### Learn about this topic:")
                st.write(st.session_state.system_response)

        # Evaluation Form
        if hasattr(st.session_state, 'system_response') and st.session_state.system_response:
            with st.form(key='evaluation_form'):
                st.markdown("### Your Understanding")
                user_response = st.text_area(
                    "Write what you understood:",
                    help="Explain the concept in your own words. Copying text will result in a score of 0."
                )
                submit_evaluation = st.form_submit_button("Evaluate Understanding")

                if submit_evaluation and user_response:
                    evaluation = evaluate_response(st.session_state.system_response, user_response)

                    if isinstance(evaluation, dict):
                        score = evaluation.get("Score", 0.000)

                        # Save score if not plagiarized
                        if evaluation.get("Category") != "0":
                            st.session_state.scores.append({
                                'timestamp': datetime.now(),
                                'score': score,
                                'question': learning_question
                            })

                            # Save evaluation history
                            st.session_state.evaluation_history.append({
                                'timestamp': datetime.now(),
                                'question': learning_question,
                                'score': score,
                                'category': evaluation.get("Category"),
                                'explanation': evaluation.get("Explanation")
                            })

                        # Display evaluation results
                        st.markdown("### Evaluation Results")
                        st.write(evaluation.get("Explanation"))
                        st.write(f"**Score:** {score:.3f}/100")
                        st.markdown("### Improvement Feedback")
                        st.write(evaluation.get("Improvement"))

                        # Display previous attempts if any
                        if len(st.session_state.evaluation_history) > 1:
                            st.markdown("### Previous Attempts")
                            history_df = pd.DataFrame(st.session_state.evaluation_history[:-1])
                            st.dataframe(history_df[['timestamp', 'question', 'score', 'explanation']])
                    else:
                        st.error(evaluation)


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
