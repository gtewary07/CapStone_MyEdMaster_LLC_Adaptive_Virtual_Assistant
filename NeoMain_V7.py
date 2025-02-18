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


############################################
# Utility Functions for Evaluation & Feedback
############################################

def check_response_similarity(system_response, user_response):
    """Check if user response is copied from system response"""
    system_clean = " ".join(system_response.lower().split())
    user_clean = " ".join(user_response.lower().split())
    if system_clean == user_clean:
        return True
    system_words = set(system_clean.split())
    user_words = set(user_clean.split())
    similarity = len(system_words.intersection(user_words)) / len(system_words)
    return similarity > 0.8


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
        "1": f"""
✅ Strong Understanding (Score: {score:.3f}/100):
- Good grasp of core concepts.
To improve further:
- Add more specific examples from your experience.
- Explain practical applications.
- Connect concepts to real-world scenarios.
""",
        "2": f"""
📝 Basic Understanding (Score: {score:.3f}/100):
To improve your response:
- Include missing key concepts.
- Provide more detailed technical explanations.
- Add examples from your professional experience.
- Focus on system components and their relationships.
""",
        "3": f"""
🌟 Advanced Understanding (Score: {score:.3f}/100):
Excellent technical knowledge! To perfect your response:
- Structure your answer more clearly.
- Prioritize the most relevant information.
- Consider adding industry-specific examples.
- Connect concepts to your professional experience.
""",
        "4": f"""
⚠️ Needs Improvement (Score: {score:.3f}/100):
To correct your response:
- Review the fundamental concepts.
- Focus on accuracy of technical details.
- Avoid assumptions.
- Start with basic definitions.
- Connect ideas logically.
"""
    }
    return feedback.get(category, "Please provide a valid response to receive improvement feedback.")


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


def extract_rationale_procedure_facts(user_response):
    """
    Naively extract rationale, procedure, and facts from user response.
    """
    response_lower = user_response.lower()

    # Check if the response contains rationale-related keywords
    if "because" in response_lower or "reason" in response_lower:
        rationale = "Identified a rationale statement (e.g., 'because', 'reason')."
    else:
        rationale = "No rationale detected."

    # Check if the response contains procedure-related keywords
    if "steps" in response_lower or "procedure" in response_lower or "how to" in response_lower:
        procedure = "Identified a procedure/method statement."
    else:
        procedure = "No procedure detected."

    # Check if the response contains factual information keywords
    if "fact" in response_lower or "data" in response_lower or "statistic" in response_lower:
        facts = "Identified factual information based on 'fact', 'data', or 'statistic'."
    else:
        facts = "No facts detected."
    return rationale, procedure, facts


def evaluate_response(system_answer, user_response):
    """Enhanced evaluation with plagiarism check, improvement feedback,
       plus rationale, procedure, facts, and understanding level (1 to 5).
    """

    # Check if the response is too similar to the system answer (possible plagiarism)
    if check_response_similarity(system_answer, user_response):
        return {
            "Category": "0",
            "Score": 0.000,
            "Explanation": "🚫 Response appears to be copied from the system answer.",
            "Improvement": generate_improvement_feedback("0", 0.000),
            "Rationale": "",
            "Procedure": "",
            "Facts": "",
            "UnderstandingLevel": 0
        }

    # Extract rationale, procedure, and facts from the response
    rationale, procedure, facts = extract_rationale_procedure_facts(user_response)
    try:
        # Assign score and category based on detected keywords
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

        # Mapping category to understanding level
        category_to_level = {
            "0": 0,
            "1": 4,
            "2": 2,
            "3": 5,
            "4": 1
        }
        understanding_level = category_to_level.get(category, 1)
        return {
            "Category": category,
            "Score": score,
            "Explanation": explanation,
            "Improvement": generate_improvement_feedback(category, score),
            "Rationale": rationale,
            "Procedure": procedure,
            "Facts": facts,
            "UnderstandingLevel": understanding_level
        }
    except Exception as e:
        return f"Evaluation error: {str(e)}"


############################################
# Functions for LLM Interaction and Quiz Generation
############################################

def get_groq_response(prompt, age, interest, experience, conversation_history=None, retries=3, delay=2):
    """
    Get response from Groq with optional conversation chaining.
    If conversation_history is provided, the new user message is appended.
    Otherwise, a new conversation is started with a system prompt.
    """
    if conversation_history is None:
        conversation_history = [
            {"role": "system",
             "content": f"You are a tutor for a {age} year old person interested in {interest} with {experience} experience. Adjust your explanation accordingly and provide clear, age-appropriate information that aligns with their professional background."}
        ]
    conversation_history.append({"role": "user", "content": prompt})
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=conversation_history,
                model="llama-3.3-70b-versatile"
            )
            assistant_message = chat_completion.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, conversation_history
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return f"An error occurred: {str(e)}", conversation_history


def generate_assessment_questions(content):
    """
    Use the LLM to generate a short multiple-choice quiz (3 questions)
    based on the provided content. The output is expected to be a JSON array
    where each element is a dict with keys: 'question', 'options', and 'answer'.
    """
    prompt = (
        f"Generate 3 multiple choice questions to assess understanding of the following content. "
        f"Format your response as a JSON array where each element is an object with keys 'question', "
        f"'options' (a list of answer choices), and 'answer' (the correct answer, which should match one of the options). "
        f"Content: {content}"
    )
    response, _ = get_groq_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
        st.session_state.user_data['experience'],
        conversation_history=None  # fresh conversation for quiz generation
    )
    # Extract the JSON portion from the response
    start_index = response.find('[')
    end_index = response.rfind(']')
    if start_index != -1 and end_index != -1:
        json_string = response[start_index:end_index + 1]
    else:
        json_string = response
    try:
        questions = json.loads(json_string)
        if not isinstance(questions, list):
            questions = []
    except Exception as e:
        st.error("Error parsing quiz questions. The response was: " + response)
        questions = []
    return questions


def get_question_explanation(question, correct_answer):
    """
    Get an in-depth explanation from the LLM as to why the given correct answer
    is correct for the provided question.
    """
    prompt = (
        f"Provide an in-depth explanation for the following question:\n\n"
        f"Question: {question}\n\n"
        f"Explain why the correct answer is '{correct_answer}' and highlight the key points that support it."
    )
    explanation, _ = get_groq_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
        st.session_state.user_data['experience'],
        conversation_history=None
    )
    return explanation


############################################
# Session State & User Data Functions
############################################

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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []


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


############################################
# Application UI: Login & Main Interface
############################################

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
                    # Uncomment the next line if your Streamlit version supports st.experimental_rerun()
                    # st.experimental_rerun()
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
    st.title(f"Welcome {st.session_state.user_data.get('interest', 'Enthusiast')}!")
    st.sidebar.header("Your Profile")
    st.sidebar.write(f"Interest: {st.session_state.user_data.get('interest', '')}")
    st.sidebar.write(f"Experience: {st.session_state.user_data.get('experience', '')}")
    tab1, tab2 = st.tabs(["Q&A Mode", "Learning Assessment Mode"])

    ############################################
    # Q&A Mode: Ask Questions, Follow-ups, and In-depth Explanations
    ############################################
    with tab1:
        st.header("Ask Questions")
        with st.form(key='qa_form'):
            question = st.text_input("Enter your question:")
            submit_qa = st.form_submit_button("Get Answer")
            if submit_qa and question:
                conversation_history = st.session_state.chat_history if st.session_state.chat_history else None
                response, conversation_history = get_groq_response(
                    question,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['interest'],
                    st.session_state.user_data['experience'],
                    conversation_history
                )
                st.session_state.chat_history = conversation_history
                st.markdown("### Answer")
                st.write(response)
        if st.session_state.chat_history:
            with st.form(key='followup_form'):
                followup_question = st.text_input("Enter a follow-up question (optional):")
                submit_followup = st.form_submit_button("Submit Follow-up")
                if submit_followup and followup_question:
                    response, conversation_history = get_groq_response(
                        followup_question,
                        st.session_state.user_data['age'],
                        st.session_state.user_data['interest'],
                        st.session_state.user_data['experience'],
                        st.session_state.chat_history
                    )
                    st.session_state.chat_history = conversation_history
                    st.markdown("### Follow-up Answer")
                    st.write(response)
            with st.form(key='in_depth_form'):
                highlighted_text = st.text_area(
                    "Paste the text (from the answer above) you want more details on (simulate highlighting):")
                submit_in_depth = st.form_submit_button("Get In-depth Explanation")
                if submit_in_depth and highlighted_text:
                    prompt = f"Provide a more in-depth explanation on the following text: {highlighted_text}"
                    in_depth_response, _ = get_groq_response(
                        prompt,
                        st.session_state.user_data['age'],
                        st.session_state.user_data['interest'],
                        st.session_state.user_data['experience'],
                        conversation_history=None
                    )
                    st.markdown("#### In-depth Explanation")
                    st.write(in_depth_response)

    ############################################
    # Learning Assessment Mode: Learning Topic, Quiz & Text Evaluation
    ############################################
    with tab2:
        st.header("Learning Assessment")

        # Form to set the learning topic
        with st.form(key='learning_form'):
            learning_question = st.text_input("What would you like to learn about?")
            submit_learning = st.form_submit_button("Start Learning")
            if submit_learning and learning_question:
                system_response, _ = get_groq_response(
                    learning_question,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['interest'],
                    st.session_state.user_data['experience'],
                    conversation_history=None
                )
                st.session_state.system_response = system_response
                st.success("Learning topic loaded!")

        # Display the system response if available; otherwise prompt the user
        if st.session_state.system_response:
            st.markdown("### Review the Topic")
            st.write(st.session_state.system_response)

            st.markdown("### Alternative: Quiz Assessment")
            if st.button("Generate Quiz Questions"):
                quiz_questions = generate_assessment_questions(st.session_state.system_response)
                st.session_state.quiz_questions = quiz_questions
            if st.session_state.quiz_questions:
                st.subheader("Answer the following questions:")
                user_answers = {}
                for idx, question_data in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{idx + 1}: {question_data['question']}**")
                    option = st.radio("Choose your answer:", question_data['options'], key=f"quiz_{idx}")
                    user_answers[idx] = option
                if st.button("Submit Quiz"):
                    correct_count = 0
                    st.markdown("### Quiz Results")
                    for idx, question_data in enumerate(st.session_state.quiz_questions):
                        correct_answer = question_data['answer']
                        user_answer = user_answers.get(idx)
                        if user_answer == correct_answer:
                            correct_count += 1
                            st.write(f"Question {idx + 1}: Correct!")
                        else:
                            st.write(f"Question {idx + 1}: Incorrect. The correct answer is: {correct_answer}")
                            explanation = get_question_explanation(question_data['question'], correct_answer)
                            st.write("Explanation:", explanation)
                    score = (correct_count / len(st.session_state.quiz_questions)) * 100
                    st.write(f"**Your Quiz Score:** {score:.1f}%")
        else:
            st.info("Please click 'Start Learning' by entering a topic above to get content for assessment.")

        # Optional: Traditional text-based evaluation of the topic
        with st.form(key='evaluation_form'):
            st.markdown("### Your Understanding (Text Response)")
            user_response = st.text_area(
                "Write what you understood:",
                help="Explain the concept in your own words. Copying text will result in a score of 0."
            )
            submit_evaluation = st.form_submit_button("Evaluate Understanding")
            if submit_evaluation and user_response:
                evaluation = evaluate_response(st.session_state.system_response, user_response)
                if isinstance(evaluation, dict):
                    score = evaluation.get("Score", 0.000)
                    understanding_level = evaluation.get("UnderstandingLevel", 0)
                    st.session_state.scores.append({
                        'timestamp': datetime.now(),
                        'score': score,
                        'question': "Learning Assessment"
                    })
                    st.session_state.evaluation_history.append({
                        'timestamp': datetime.now(),
                        'question': "Learning Assessment",
                        'score': score,
                        'category': evaluation.get("Category"),
                        'explanation': evaluation.get("Explanation"),
                        'level': understanding_level
                    })
                    st.markdown("### Evaluation Results")
                    st.write(evaluation.get("Explanation"))
                    st.write(f"**Score:** {score:.3f}/100")
                    st.write(f"**Understanding Level:** {understanding_level}/5")
                    st.markdown("### Improvement Feedback")
                    st.write(evaluation.get("Improvement"))
                    st.markdown("### Parameters Breakdown")
                    st.write(f"**Rationale:** {evaluation.get('Rationale')}")
                    st.write(f"**Procedure:** {evaluation.get('Procedure')}")
                    st.write(f"**Facts:** {evaluation.get('Facts')}")
                    if len(st.session_state.evaluation_history) > 1:
                        st.markdown("### Previous Attempts")
                        history_df = pd.DataFrame(st.session_state.evaluation_history[:-1])
                        st.dataframe(history_df[['timestamp', 'question', 'score', 'explanation', 'level']])
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
            # Uncomment the next line if your Streamlit version supports st.experimental_rerun()
            # st.experimental_rerun()


if __name__ == "__main__":
    main()
