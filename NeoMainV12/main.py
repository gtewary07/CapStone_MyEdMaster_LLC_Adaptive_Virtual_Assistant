import os
import streamlit as st
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
import importlib.util
import io
import requests
import re
from enum import Enum




# Check for required packages
def check_dependency(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None


# Dictionary of packages and their descriptions
required_packages = {
    'streamlit': 'Required for web interface',
    'numpy': 'Required for data manipulation',
    'pandas': 'Required for data manipulation',
    'matplotlib': 'Required for chart generation',
    'reportlab': 'Required for PDF report generation (optional)',
    'PyPDF2': 'Required for PDF processing',
    'groq': 'Required for API interaction'
}

# Check which packages are missing
missing_packages = []
for package, description in required_packages.items():
    if not check_dependency(package):
        missing_packages.append(f"{package}: {description}")

# Print missing package information
if missing_packages:
    print("\n=============== DEPENDENCY WARNING ===============")
    print("Missing required packages. Please install them using pip:")
    print("pip install " + " ".join([pkg.split(':')[0] for pkg in missing_packages]))
    print("\nMissing packages and their purposes:")
    for pkg in missing_packages:
        print(f"- {pkg}")
    print("=================================================\n")


# Define LLM API options as an enum (MUST define this first)
class LLMProvider(Enum):
    GROQ = "Groq"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    CUSTOM = "Custom API"
    LOCAL = "Local Fallback (No API)"


# Conditional imports based on availability
try:
    from groq import Groq
    client = Groq(api_key="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")
except ImportError:
    pass

# Import numpy and matplotlib
import numpy as np

if check_dependency('matplotlib'):
    import matplotlib.pyplot as plt

# Import reportlab conditionally
REPORTLAB_AVAILABLE = check_dependency('reportlab')
if REPORTLAB_AVAILABLE:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# Import PyPDF2 conditionally
PYPDF2_AVAILABLE = check_dependency('PyPDF2')
if PYPDF2_AVAILABLE:
    import PyPDF2

# Import from other files
from assessment import (
    get_question_explanation, extract_text_from_pdf
)
from report import (
    generate_understanding_report, display_learning_report,
    create_understanding_radar_chart, create_progress_chart,
    generate_pdf_report
)



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
    if 'direct_questions' not in st.session_state:
        st.session_state.direct_questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'direct_answer_evaluations' not in st.session_state:
        st.session_state.direct_answer_evaluations = []
    if 'topic_complexity' not in st.session_state:
        st.session_state.topic_complexity = {}
    if 'understanding_report' not in st.session_state:
        st.session_state.understanding_report = {}
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'learning_chat_history' not in st.session_state:
        st.session_state.learning_chat_history = []
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""
    # New session state variables
    if 'difficulty_level' not in st.session_state:
        st.session_state.difficulty_level = "Basic"  # Always start with Basic for new users
    if 'adaptive_mode' not in st.session_state:
        st.session_state.adaptive_mode = True  # Enable adaptive mode by default
    if 'previous_assessment_results' not in st.session_state:
        st.session_state.previous_assessment_results = {}
    if 'is_first_login' not in st.session_state:
        st.session_state.is_first_login = True
    if 'post_assessment_feedback' not in st.session_state:
        st.session_state.post_assessment_feedback = ""
    if 'adaptive_learning_areas' not in st.session_state:
        st.session_state.adaptive_learning_areas = []
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = LLMProvider.GROQ
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            LLMProvider.GROQ.value: "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",
            LLMProvider.OPENAI.value: "",
            LLMProvider.ANTHROPIC.value: "",
            LLMProvider.CUSTOM.value: ""
        }
    if 'custom_api_url' not in st.session_state:
        st.session_state.custom_api_url = ""
    if 'groq_model' not in st.session_state:
        st.session_state.groq_model = "llama-3.1-70b-versatile"  # Default model
    if st.session_state.logged_in:
        # Add the learning report uploader before the learning form
        add_learning_report_uploader()


def save_user_data(username, password, age, experience):
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
# LLM and API Functions
############################################

def get_llm_client():
    """Get an appropriate client for the selected LLM provider"""
    provider = st.session_state.llm_provider

    if provider == LLMProvider.GROQ:
        try:
            from groq import Groq
            return Groq(api_key=st.session_state.api_keys.get(LLMProvider.GROQ.value, ""))
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            return None

    elif provider == LLMProvider.OPENAI:
        try:
            import openai
            return openai.OpenAI(api_key=st.session_state.api_keys.get(LLMProvider.OPENAI.value, ""))
        except ImportError:
            st.error("OpenAI package not installed. Run: pip install openai")
            return None
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None

    elif provider == LLMProvider.ANTHROPIC:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=st.session_state.api_keys.get(LLMProvider.ANTHROPIC.value, ""))
        except ImportError:
            st.error("Anthropic package not installed. Run: pip install anthropic")
            return None
        except Exception as e:
            st.error(f"Error initializing Anthropic client: {str(e)}")
            return None

    elif provider == LLMProvider.CUSTOM:
        # Return a placeholder for custom API - will need custom handling in get_response
        return {
            "type": "custom",
            "api_key": st.session_state.api_keys.get(LLMProvider.CUSTOM.value, ""),
            "url": st.session_state.custom_api_url
        }

    else:  # LOCAL
        return None


def integrate_learning_report_feature():
    """
    Integrate the learning report feature into the main application.
    Add this code to the main_application() function.
    """
    # Add the learning report uploader
    add_learning_report_uploader()

    # Check if we have post assessment feedback from a report
    if 'post_assessment_feedback' in st.session_state and st.session_state.post_assessment_feedback:
        st.markdown("### Learning Recommendations")
        st.markdown(st.session_state.post_assessment_feedback)

        if st.button("Clear Recommendations"):
            st.session_state.post_assessment_feedback = ""
            st.rerun()


def add_learning_report_uploader():
    """Add a learning report uploader section to continue learning based on previous report"""
    # Main expander for the report uploader
    with st.expander("📊 Continue Learning from Previous Report", expanded=False):
        st.markdown("""
        ### Upload a Previous Learning Report
        You can upload a learning report from a previous session to continue your learning journey. 
        The system will use information from the report to customize your learning experience.
        """)

        uploaded_report = st.file_uploader(
            "Upload Learning Report PDF",
            type="pdf",
            key="learning_report_uploader",
            help="Upload a learning report PDF generated from a previous assessment"
        )

        if uploaded_report:
            # Process the learning report
            user_data, understanding_levels, areas_for_improvement, success = process_learning_report_pdf(
                uploaded_report,
                client=client
            )

    # Debug info sections - OUTSIDE the main expander to avoid nesting
    if uploaded_report:
        # For debugging - show extracted text if available
        if 'debug_pdf_text' in st.session_state:
            show_pdf_text = st.checkbox("Show PDF Text Debug", value=False)
            if show_pdf_text:
                st.text_area("Extracted PDF Text (first 1000 chars)",
                             value=st.session_state.debug_pdf_text[:1000] + "...",
                             height=200)

        # For debugging - show error details if available
        if 'pdf_error' in st.session_state:
            show_error = st.checkbox("Show Error Details", value=False)
            if show_error:
                st.code(st.session_state.pdf_error)

        if success:
            # Display extracted information
            st.success("Learning report processed successfully!")

            # Display user profile
            st.subheader("Learner Profile")
            st.write(f"Age: {user_data.get('age', 'Not found')}")
            st.write(f"Experience: {user_data.get('experience', 'Not found')}")

            # Display understanding levels with color-coded badges
            st.subheader("Understanding Levels")

            # Helper function for level color and description
            def get_level_info(level):
                colors = ["gray", "red", "orange", "blue", "green", "purple"]
                descriptions = ["Not Assessed", "Minimal", "Basic", "Moderate", "Good", "Excellent"]
                color = colors[min(level, 5)]
                desc = descriptions[min(level, 5)]
                return color, desc

            # Display overall level with color
            overall_level = understanding_levels.get('overall_level', 0)
            color, desc = get_level_info(overall_level)
            st.markdown(
                f"<div style='padding:10px; border-radius:5px; background-color:{color}20; margin-bottom:10px;'>"
                f"<span style='font-weight:bold; font-size:1.2em;'>Overall: "
                f"<span style='color:{color};'>{overall_level}/5 - {desc}</span></span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Create columns for dimension levels
            cols = st.columns(3)
            dimensions = [
                ("Rationale (Why)", 'rationale_level'),
                ("Factual (What)", 'factual_level'),
                ("Procedural (How)", 'procedural_level')
            ]

            # Display each dimension
            for i, (name, key) in enumerate(dimensions):
                level = understanding_levels.get(key, 0)
                color, desc = get_level_info(level)
                cols[i].markdown(
                    f"<div style='padding:8px; border-radius:5px; background-color:{color}20;'>"
                    f"<span style='font-weight:bold;'>{name}:</span><br>"
                    f"<span style='color:{color};'>{level}/5 - {desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Display areas for improvement
            if areas_for_improvement:
                st.subheader("Areas for Improvement")
                for area in areas_for_improvement:
                    if area == 'rationale':
                        st.write("- **Rationale Understanding** (the 'why' behind concepts)")
                    elif area == 'factual':
                        st.write("- **Factual Knowledge** (key terms and concepts)")
                    elif area == 'procedural':
                        st.write("- **Procedural Knowledge** (how to apply concepts)")

            # Show debug info if available
            if 'debug_info' in st.session_state:
                show_debug = st.checkbox("Show Detailed Debug Info", value=False)
                if show_debug:
                    st.json(st.session_state.debug_info)

            # Apply the settings to the session
            if st.button("Apply These Learning Settings"):
                # Update adaptive learning areas
                st.session_state.adaptive_learning_areas = areas_for_improvement

                # Update user data if applicable
                if user_data:
                    # Only update if not already set
                    if 'user_data' not in st.session_state or not st.session_state.user_data:
                        st.session_state.user_data = user_data
                    else:
                        # Optionally update age and experience
                        if 'age' in user_data:
                            st.session_state.user_data['age'] = user_data['age']
                        if 'experience' in user_data:
                            st.session_state.user_data['experience'] = user_data['experience']

                # Enable adaptive mode
                st.session_state.adaptive_mode = True

                # Set previous assessment results
                st.session_state.previous_assessment_results = {
                    'overall_level': understanding_levels.get('overall_level', 0),
                    'rationale_level': understanding_levels.get('rationale_level', 0),
                    'factual_level': understanding_levels.get('factual_level', 0),
                    'procedural_level': understanding_levels.get('procedural_level', 0)
                }

                # Clear any existing post-assessment feedback
                if 'post_assessment_feedback' in st.session_state:
                    st.session_state.post_assessment_feedback = ""

                # Generate new post-assessment feedback based on the report
                feedback = generate_post_assessment_feedback(
                    st.session_state.previous_assessment_results,
                    st.session_state.current_topic or "your topic"
                )
                st.session_state.post_assessment_feedback = feedback

                st.success(
                    "Learning settings applied! Your next learning session will be customized based on these levels.")
                st.info("Start a new learning session with the 'Start Learning' button to see personalized content.")
                st.rerun()  # Refresh the page to show the feedback
        else:
            st.error(
                "Could not process the learning report. Please make sure you're uploading a valid learning report PDF.")

def generate_fallback_response(topic, difficulty_level="Basic"):
    """
    Generate a fallback response if the API fails.
    This provides basic content so the application can continue functioning.
    """
    difficulty_descriptions = {
        "Basic": "introduces fundamental concepts in simple terms",
        "Intermediate": "covers more detailed aspects with some technical terms",
        "Advanced": "explores complex technical details and sophisticated analysis"
    }

    return f"""
# Introduction to {topic}

This is a {difficulty_level.lower()}-level overview of {topic}. This content {difficulty_descriptions.get(difficulty_level, '')}.

## Key Concepts

Due to a technical issue with our content generation system, we couldn't provide a full response. Here are some general learning strategies you can use while exploring this topic:

1. **Start with the fundamentals**: Make sure you understand the basic principles before moving to more advanced topics.
2. **Connect concepts**: Try to see how different ideas relate to each other.
3. **Apply knowledge**: Practice applying what you learn to real-world situations.
4. **Ask questions**: Critical thinking involves questioning assumptions and exploring different perspectives.

## Learning Resources

You may want to explore other resources on this topic:
- Online courses and tutorials
- Textbooks and academic papers
- Video lectures and demonstrations
- Interactive learning platforms

*Note: This is a fallback response due to a technical issue. Please try again later for a complete educational response.*
"""


def get_llm_response(prompt, age, experience, conversation_history=None, client=None, retries=3, delay=2):
    """
    Get response from the selected LLM provider.
    This function replaces get_groq_response and supports multiple providers.
    """
    if conversation_history is None:
        # Enhanced system prompt with more specific instructions
        system_prompt = (
            f"You are an intelligent, patient, and engaging tutor for a {age} year old person "
            f"interested in {experience} experience. Your responses should be:"
            f"\n1. Clear and age-appropriate, using analogies and examples suited to the user's background"
            f"\n2. Well-structured with headings and sections for complex topics"
            f"\n3. Engaging, with questions to check understanding where appropriate"
            f"\n4. Comprehensive but not overwhelming, with a balance of depth and accessibility"
            f"\n5. Connected to the user's professional background  {experience})"
            f"\n6. Educational in tone, focusing on building understanding rather than just providing information"
            f"\n7. Include practical applications and real-world examples where possible"
        )
        conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

    conversation_history.append({"role": "user", "content": prompt})

    # Check if we're in local fallback mode
    if st.session_state.llm_provider == LLMProvider.LOCAL:
        # Generate a simple response based on the prompt
        response = generate_fallback_response(prompt.split(":")[-1] if ":" in prompt else prompt)
        conversation_history.append({"role": "assistant", "content": response})
        return response, conversation_history

    # If client is None, get a new client
    if client is None:
        client = get_llm_client()
        if client is None and st.session_state.llm_provider != LLMProvider.LOCAL:
            st.error(f"Failed to initialize {st.session_state.llm_provider.value} client. Check API key and try again.")
            response = generate_fallback_response(prompt.split(":")[-1] if ":" in prompt else prompt)
            conversation_history.append({"role": "assistant", "content": response})
            return response, conversation_history

    last_error = None

    # Handle different provider types
    if st.session_state.llm_provider == LLMProvider.GROQ:
        # List of models to try in order of preference
        models_to_try = [
            st.session_state.groq_model,  # Try the remembered working model first
            "llama-3.3-70b-versatile",  # From the table - Meta's latest model
            "llama-3.1-8b-instant",  # A smaller, faster alternative
            "llama3-70b-8192",  # Another viable option
            "llama3-8b-8192",  # Smaller variant
            "mixtral-8x7b-32768"  # Keep this as a fallback
        ]

        # Remove duplicates while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for model in models_to_try:
            for attempt in range(retries):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=conversation_history,
                        model=model,
                        temperature=0.7,
                        max_tokens=2048
                    )
                    assistant_message = chat_completion.choices[0].message.content
                    conversation_history.append({"role": "assistant", "content": assistant_message})

                    # If successful with this model, remember it for future use
                    if st.session_state.groq_model != model:
                        st.session_state.groq_model = model
                        st.info(f"Using Groq model: {model}")

                    return assistant_message, conversation_history
                except Exception as e:
                    last_error = e
                    if "404" in str(e) and attempt == retries - 1:
                        # This model doesn't exist, try the next one
                        break
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue

    elif st.session_state.llm_provider == LLMProvider.OPENAI:
        try:
            chat_completion = client.chat.completions.create(
                messages=conversation_history,
                model="gpt-3.5-turbo",  # Default model
                temperature=0.7,
                max_tokens=2048
            )
            assistant_message = chat_completion.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, conversation_history
        except Exception as e:
            last_error = e

    elif st.session_state.llm_provider == LLMProvider.ANTHROPIC:
        try:
            # Convert conversation history format for Anthropic
            messages = []
            system = ""
            for msg in conversation_history:
                if msg["role"] == "system":
                    # Anthropic uses a separate system parameter
                    system = msg["content"]
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                system=system,
                messages=messages,
                max_tokens=2048
            )
            assistant_message = response.content[0].text
            conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, conversation_history
        except Exception as e:
            last_error = e

    elif st.session_state.llm_provider == LLMProvider.CUSTOM:
        try:
            # Custom API implementation
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {client['api_key']}"
            }

            payload = {
                "messages": conversation_history,
                "temperature": 0.7,
                "max_tokens": 2048
            }

            response = requests.post(
                client['url'],
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # Assumes a standard format similar to OpenAI
                assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message, conversation_history
            else:
                last_error = f"API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            last_error = e

    # If we get here, all attempts failed
    error_message = f"Failed to get a response from {st.session_state.llm_provider.value} API. Error: {str(last_error)}"
    st.error(error_message)

    # Return a fallback response
    response = generate_fallback_response(prompt.split(":")[-1] if ":" in prompt else prompt)
    conversation_history.append({"role": "assistant", "content": response})
    return response, conversation_history


def determine_topic_complexity(content, client):
    """
    Analyze the content to determine its complexity and suggest an appropriate number of questions.
    Returns a complexity level (1-5) and recommended number of questions.
    """
    try:
        prompt = (
            f"Analyze the following educational content and determine its complexity on a scale of 1-5, "
            f"where 1 is very simple and 5 is very complex. Based on the complexity, suggest an appropriate "
            f"number of questions to test understanding (2-3 questions for simple topics, 4-6 for moderately "
            f"complex topics, and 7-10 for very complex topics). Also, suggest a distribution of question types "
            f"(rationale, factual, procedural) that would be appropriate for this content. "
            f"Return your response in valid JSON format with keys 'complexity_level', 'recommended_questions', and "
            f"'question_distribution' (an object with keys 'rationale', 'factual', 'procedural' and values as percentages). "
            f"Make sure to use proper JSON syntax with double quotes around property names and string values. "
            f"Content: {content[:2000]}"  # Limit content length further to avoid token issues
        )

        # First, try to get a response from the API
        try:
            response, _ = get_llm_response(
                prompt,
                st.session_state.user_data['age'],
                st.session_state.user_data['experience'],
                conversation_history=None,
                client=client
            )
        except Exception as api_error:
            st.error(f"Error calling the LLM API: {str(api_error)}")
            # Return default values if API call fails
            return {
                'complexity_level': 3,
                'recommended_questions': 5,
                'question_distribution': {'rationale': 33, 'factual': 33, 'procedural': 34}
            }

        # Try multiple methods to extract valid JSON from the response
        json_methods = [
            # Method 1: Try to find JSON between { and }
            lambda r: r[r.find('{'):r.rfind('}') + 1] if r.find('{') != -1 and r.rfind('}') != -1 else None,

            # Method 2: Try to find JSON between code blocks
            lambda r: r[r.find('```json') + 7:r.rfind('```')] if r.find('```json') != -1 and r.rfind(
                '```') != -1 else None,

            # Method 3: Try to find JSON between any code blocks
            lambda r: r[r.find('```') + 3:r.rfind('```')] if r.find('```') != -1 and r.rfind('```') != -1 else None,
        ]

        for method in json_methods:
            try:
                json_string = method(response)
                if json_string:
                    analysis = json.loads(json_string)
                    if isinstance(analysis, dict):
                        return {
                            'complexity_level': int(analysis.get('complexity_level', 3)),
                            'recommended_questions': int(analysis.get('recommended_questions', 5)),
                            'question_distribution': analysis.get('question_distribution',
                                                                  {'rationale': 33, 'factual': 33, 'procedural': 34})
                        }
            except Exception:
                # Continue to the next method if this one fails
                continue

        # If all JSON parsing methods fail, try to extract values directly
        try:
            # Try to find complexity level
            complexity_match = re.search(r'complexity_level["\']?\s*[:=]\s*(\d+)', response)
            complexity_level = int(complexity_match.group(1)) if complexity_match else 3

            # Try to find recommended questions
            questions_match = re.search(r'recommended_questions["\']?\s*[:=]\s*(\d+)', response)
            recommended_questions = int(questions_match.group(1)) if questions_match else 5

            return {
                'complexity_level': complexity_level,
                'recommended_questions': recommended_questions,
                'question_distribution': {'rationale': 33, 'factual': 33, 'procedural': 34}
            }
        except Exception:
            # If all methods fail, return default values
            st.warning("Could not parse complexity analysis response. Using default values.")
            return {
                'complexity_level': 3,
                'recommended_questions': 5,
                'question_distribution': {'rationale': 33, 'factual': 33, 'procedural': 34}
            }

    except Exception as e:
        st.error(f"Error analyzing topic complexity: {str(e)}")
        return {
            'complexity_level': 3,
            'recommended_questions': 5,
            'question_distribution': {'rationale': 33, 'factual': 33, 'procedural': 34}
        }


def generate_assessment_questions(content, client):
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
    response, _ = get_llm_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['experience'],
        conversation_history=None,  # fresh conversation for quiz generation
        client=client
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


def generate_direct_questions(content, num_questions=5, question_distribution=None, client=None):
    """
    Generate a set of direct questions (not multiple choice) to assess understanding
    of the provided content. Returns a list of dictionaries with question, expected_answer,
    and question_type (rationale, factual, or procedural).
    """
    if question_distribution is None:
        question_distribution = {'rationale': 33, 'factual': 33, 'procedural': 34}

    # Calculate number of questions of each type
    rationale_count = max(1, round(num_questions * question_distribution['rationale'] / 100))
    factual_count = max(1, round(num_questions * question_distribution['factual'] / 100))
    procedural_count = max(1, num_questions - rationale_count - factual_count)

    prompt = (
        f"Based on the following content, generate {num_questions} specific questions to assess "
        f"understanding of key concepts. Questions should be distributed across these types:\n"
        f"- {rationale_count} Rationale questions (why something happens, reasons, causes, etc.)\n"
        f"- {factual_count} Factual questions (definitions, dates, names, specific information)\n"
        f"- {procedural_count} Procedural questions (how to do something, steps, methods)\n\n"
        f"For each question, provide an expected answer or key points that should be included in a good answer. "
        f"Format your response as a JSON array where each element is an object with keys 'question', "
        f"'expected_answer', and 'question_type' (must be 'rationale', 'factual', or 'procedural'). "
        f"Content: {content[:4000]}"  # Limit content length to avoid token issues
    )

    response, _ = get_llm_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['experience'],
        conversation_history=None,  # fresh conversation for questions generation
        client=client
    )

    try:
        # First, check if the response contains markdown code blocks and extract the content
        if '```' in response:
            # Extract content between code blocks
            start_index = response.find('```') + 3
            # Find the language identifier (if any) and skip to the next line
            if start_index < len(response) and response[start_index:].find('\n') != -1:
                start_index = response.find('\n', start_index) + 1
            end_index = response.rfind('```')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_string = response[start_index:end_index].strip()
            else:
                # If we can't properly extract from code blocks, try normal JSON extraction
                start_index = response.find('[')
                end_index = response.rfind(']')
                if start_index != -1 and end_index != -1:
                    json_string = response[start_index:end_index + 1]
                else:
                    json_string = response
        else:
            # Try to extract just the JSON array if there are no code blocks
            start_index = response.find('[')
            end_index = response.rfind(']')
            if start_index != -1 and end_index != -1:
                json_string = response[start_index:end_index + 1]
            else:
                json_string = response

        # Parse the JSON
        questions = json.loads(json_string)
        if not isinstance(questions, list):
            questions = []

        # Validate and clean up the questions
        valid_questions = []
        for q in questions:
            if all(k in q for k in ['question', 'expected_answer', 'question_type']):
                if q['question_type'] not in ['rationale', 'factual', 'procedural']:
                    q['question_type'] = 'factual'  # Default to factual if invalid type
                valid_questions.append(q)

        return valid_questions
    except Exception as e:
        st.error(f"Error parsing direct questions: {str(e)}. The response was: {response}")

        # Create fallback questions
        return [
            {
                "question": "What is the main concept discussed in this content?",
                "expected_answer": "The main concept is about understanding the key elements of the topic.",
                "question_type": "factual"
            },
            {
                "question": "Why is this topic important?",
                "expected_answer": "This topic is important because it helps us understand fundamental principles.",
                "question_type": "rationale"
            },
            {
                "question": "How would you apply this knowledge in a real situation?",
                "expected_answer": "You would apply this knowledge by following specific steps and procedures.",
                "question_type": "procedural"
            }
        ]


def evaluate_direct_answer(user_answer, expected_answer, question_type=None, client=None):
    """
    Evaluate a user's answer to a direct question against the expected answer.
    Returns a level (1-5) and feedback.

    Parameters:
    user_answer (dict): Dictionary containing the 'question' and 'answer'
    expected_answer (str): The expected answer
    question_type (str, optional): Type of question ('rationale', 'factual', or 'procedural')
                                  Defaults to 'factual' if not provided
    """
    # Ensure question_type has a value
    if question_type is None:
        question_type = 'factual'

    prompt = (
        f"Evaluate this user's answer against the expected answer for a {question_type} question. "
        f"Provide an understanding level from 1-5 (where 1 is minimal understanding and 5 is excellent understanding). "
        f"Also provide specific feedback on what was good and what could be improved. "
        f"Format your response as a JSON object with keys 'understanding_level' (integer 1-5) and 'feedback' (string). "
        f"\n\nQuestion: {user_answer['question']}\n\n"
        f"Expected Answer: {expected_answer}\n\n"
        f"User's Answer: {user_answer['answer']}"
    )

    try:
        response, _ = get_llm_response(
            prompt,
            st.session_state.user_data['age'],
            st.session_state.user_data['experience'],
            conversation_history=None,
            client=client
        )

        # Try to extract just the JSON part if there's extra text
        start_index = response.find('{')
        end_index = response.rfind('}')
        if start_index != -1 and end_index != -1:
            json_string = response[start_index:end_index + 1]
            evaluation = json.loads(json_string)
        else:
            evaluation = json.loads(response)

        understanding_level = evaluation.get("understanding_level", 1)

        # Add a 'score' key for backward compatibility with any legacy code
        result = {
            "understanding_level": understanding_level,
            "feedback": evaluation.get("feedback", "No feedback provided."),
            "score": understanding_level * 20  # Convert 1-5 scale to 0-100 scale for compatibility
        }

        return result
    except Exception as e:
        st.error(f"Error parsing evaluation: {str(e)}")
        return {
            "understanding_level": 1,
            "feedback": "Error evaluating answer. Please try again.",
            "score": 20  # Default score for error case (1 * 20)
        }


############################################
# PDF Processing Functions
############################################

def process_uploaded_pdf(uploaded_file, difficulty_level="Basic"):
    """Process uploaded PDF file and return the extracted text with difficulty-adjusted summary"""
    if uploaded_file is not None:
        try:
            # Check if PyPDF2 is available
            if not PYPDF2_AVAILABLE:
                st.error("PyPDF2 library is not installed. PDF processing is not available.")
                st.info("To install PyPDF2, run: pip install PyPDF2")
                return "", "Error: PyPDF2 library is not installed. PDF processing is not available."

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Generate summary with difficulty level consideration
            summary_prompt = (
                f"Generate a comprehensive {difficulty_level.lower()}-level summary of the following text extracted from a PDF. "
                f"If '{difficulty_level}' is 'Basic', focus on fundamental concepts with simple explanations and minimal technical terms. "
                f"If '{difficulty_level}' is 'Intermediate', include more detailed explanations and some technical terms. "
                f"If '{difficulty_level}' is 'Advanced', provide in-depth technical details and sophisticated analysis. "
                f"Focus on the main concepts, key points, and important details. "
                f"Make the summary educational and suitable for learning purposes:\n\n{pdf_text[:10000]}"
                # Limit text length
            )

            summary, _ = get_llm_response(
                summary_prompt,
                st.session_state.user_data['age'],
                st.session_state.user_data['experience'],
                conversation_history=None,
                client=client
            )

            return pdf_text, summary
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return "", f"Error generating summary: {str(e)}"
    return "", ""


############################################
# Adaptive Learning Functions
############################################

def adapt_content_based_on_assessment(topic, previous_results=None):
    """
    Adjust content difficulty and focus based on previous assessment results
    Returns tuple of (adjusted_difficulty, focus_instruction, focus_area, focus_level)
    """
    if previous_results is None or not st.session_state.adaptive_mode:
        return st.session_state.difficulty_level, "", None, 0

    # Extract understanding levels for different dimensions
    overall_level = previous_results.get('overall_level', 0)
    rationale_level = previous_results.get('rationale_level', 0)
    factual_level = previous_results.get('factual_level', 0)
    procedural_level = previous_results.get('procedural_level', 0)

    # Determine appropriate difficulty level
    if overall_level <= 2:
        adjusted_difficulty = "Basic"
    elif overall_level == 3:
        adjusted_difficulty = "Intermediate"
    else:
        adjusted_difficulty = "Advanced"

    # Identify areas to focus on (areas with lowest scores)
    dimensions = [
        ("rationale", rationale_level),
        ("factual", factual_level),
        ("procedural", procedural_level)
    ]

    # Sort dimensions by level (ascending)
    dimensions.sort(key=lambda x: x[1])
    focus_area = dimensions[0][0] if dimensions[0][1] < 4 else None
    focus_level = dimensions[0][1] if dimensions[0][1] < 4 else 0

    # Create focus instruction
    focus_instruction = ""
    if focus_area:
        if focus_area == "rationale":
            focus_instruction = "Put extra emphasis on explaining the 'why' behind concepts - focus on reasons, causes, and implications."
        elif focus_area == "factual":
            focus_instruction = "Include more foundational facts, definitions, and key terms - focus on building a strong factual base."
        elif focus_area == "procedural":
            focus_instruction = "Include more step-by-step procedures, examples, and practical applications - focus on how to apply concepts."

    return adjusted_difficulty, focus_instruction, focus_area, focus_level


def generate_post_assessment_feedback(report, topic):
    """
    Generate a helpful feedback response after assessment to guide the user on what to focus on next
    """
    if not report:
        return ""

    overall_level = report.get('overall_level', 0)
    rationale_level = report.get('rationale_level', 0)
    factual_level = report.get('factual_level', 0)
    procedural_level = report.get('procedural_level', 0)

    # Identify the weakest area
    dimensions = [
        ("understanding the reasons and causes", rationale_level, "rationale"),
        ("grasping key facts and concepts", factual_level, "factual"),
        ("applying concepts practically", procedural_level, "procedural")
    ]

    # Sort by level (ascending)
    dimensions.sort(key=lambda x: x[1])
    weakest_area = dimensions[0]

    # Store the weakest areas for future reference
    st.session_state.adaptive_learning_areas = [dim[2] for dim in dimensions if dim[1] < 4]

    # Generate feedback
    feedback = f"""
## Assessment Feedback - {topic}

Based on your assessment results, you've demonstrated a level {overall_level}/5 understanding of this topic.

### Areas to Focus On

Your responses indicate that you could benefit from additional focus on **{weakest_area[0]}** (level {weakest_area[1]}/5).

### Recommendations:
"""

    # Add specific recommendations
    if weakest_area[2] == "rationale":
        feedback += """
- Try to answer "why" questions about the topic
- Look for cause-and-effect relationships
- Connect concepts to underlying principles
- Ask yourself "Why does this happen?" when learning new facts
"""
    elif weakest_area[2] == "factual":
        feedback += """
- Review the fundamental terminology and definitions
- Create flashcards for key concepts
- Practice recalling specific facts about the topic
- Build a stronger foundation of core knowledge
"""
    elif weakest_area[2] == "procedural":
        feedback += """
- Work through step-by-step examples
- Apply concepts to practical scenarios
- Try to explain how to implement the knowledge
- Practice with hands-on exercises when possible
"""

    feedback += """
### Next Steps

When you continue learning about this topic, the system will automatically adjust the content to focus on these areas. Your next lesson will be tailored to address these specific learning needs.

Click "Start Learning" again to continue with an adaptive learning experience.
"""

    return feedback


def add_adaptive_mode_toggle():
    """Add adaptive learning mode toggle to the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Learning Settings")
    st.sidebar.markdown("""
    <div style="background-color: #f0f7fb; padding: 10px; border-radius: 5px; border-left: 5px solid #3498db;">
        <h4 style="margin-top: 0;">Adaptive Learning</h4>
        <p style="margin-bottom: 0;">When enabled, the system will analyze your assessment results and adjust future content to focus on areas where you need the most improvement.</p>
    </div>
    """, unsafe_allow_html=True)

    adaptive_mode = st.sidebar.checkbox(
        "Enable Adaptive Learning",
        value=st.session_state.adaptive_mode,
        help="The system will adjust content difficulty and focus based on your assessment results"
    )
    st.session_state.adaptive_mode = adaptive_mode


def add_llm_settings():
    """Add LLM API settings to the sidebar"""
    with st.sidebar.expander("🤖 LLM API Settings", expanded=False):
        # LLM Provider Selection
        provider_options = [provider.value for provider in LLMProvider]
        current_provider = st.session_state.llm_provider.value if isinstance(st.session_state.llm_provider,
                                                                             LLMProvider) else "Groq"

        selected_provider = st.radio(
            "Select LLM Provider:",
            provider_options,
            index=provider_options.index(current_provider)
        )

        # Update the provider in session state
        st.session_state.llm_provider = next(
            provider for provider in LLMProvider if provider.value == selected_provider)

        # Show appropriate API key input based on selection
        if st.session_state.llm_provider != LLMProvider.LOCAL:
            api_key = st.text_input(
                f"{st.session_state.llm_provider.value} API Key:",
                value=st.session_state.api_keys.get(st.session_state.llm_provider.value, ""),
                type="password"
            )
            st.session_state.api_keys[st.session_state.llm_provider.value] = api_key

            # Show custom API URL input for custom provider
            if st.session_state.llm_provider == LLMProvider.CUSTOM:
                custom_url = st.text_input(
                    "Custom API URL:",
                    value=st.session_state.custom_api_url
                )
                st.session_state.custom_api_url = custom_url

        if st.button("Test Connection"):
            test_llm_connection()


def process_learning_report_pdf(uploaded_file, client=None):
    """
    Process an uploaded learning report PDF file and extract relevant user data and learning metrics.
    Updated to handle reports with various formatting patterns.

    Args:
        uploaded_file: A file-like object containing PDF learning report data
        client: Optional LLM client for enhanced processing

    Returns:
        tuple: (user_data, understanding_levels, areas_for_improvement, parsed_successfully)
    """
    if uploaded_file is None:
        return {}, {}, [], False

    try:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Store for debugging
        st.session_state.debug_pdf_text = pdf_text

        # Initialize default return values
        user_data = {}
        understanding_levels = {
            'overall_level': 0,
            'rationale_level': 0,
            'factual_level': 0,
            'procedural_level': 0
        }
        areas_for_improvement = []

        # Extract report title/topic (from first line)
        title_match = re.search(r'Learning Report:\s*([^\n]+)', pdf_text)
        if title_match:
            topic = title_match.group(1).strip()
            st.session_state.current_topic = topic

        # Updated pattern for username - handle multi-line format
        username_pattern = r'User\s*\n?\s*(\w+)'
        age_pattern = r'Age\s*\n?\s*(\d+)'
        experience_pattern = r'Experience\s*\n?\s*([^\n]+?)\s*(?=Understanding|Dimension)'

        user_match = re.search(username_pattern, pdf_text)
        age_match = re.search(age_pattern, pdf_text)
        exp_match = re.search(experience_pattern, pdf_text, re.DOTALL)

        if user_match:
            user_data['username'] = user_match.group(1).strip()
        if age_match:
            try:
                user_data['age'] = int(age_match.group(1).strip())
            except ValueError:
                # Handle case where age might not be a clean integer
                age_str = age_match.group(1).strip()
                # Try to extract just the digits
                digits = ''.join(c for c in age_str if c.isdigit())
                if digits:
                    user_data['age'] = int(digits)
        if exp_match:
            user_data['experience'] = exp_match.group(1).strip()

        # Extract overall understanding level
        overall_pattern = r'Overall Understanding:\s*Level\s*(\d+)/5'
        overall_match = re.search(overall_pattern, pdf_text)
        if overall_match:
            understanding_levels['overall_level'] = int(overall_match.group(1))

        # Extract dimension levels - more flexible patterns
        dimension_patterns = {
            'rationale_level': r'Rationale\s*\(Why\)\s*\n?\s*(\d+)/5',
            'factual_level': r'Factual\s*\(What\)\s*\n?\s*(\d+)/5',
            'procedural_level': r'Procedural\s*\(How\)\s*\n?\s*(\d+)/5'
        }

        for key, pattern in dimension_patterns.items():
            match = re.search(pattern, pdf_text)
            if match:
                understanding_levels[key] = int(match.group(1))

        # Extract areas for improvement
        focus_pattern = r'Learning Focus:\s*([^\n]+)'
        focus_match = re.search(focus_pattern, pdf_text)

        if focus_match:
            focus_text = focus_match.group(1).strip().lower()
            # Check which dimension is mentioned
            if 'why' in focus_text or 'rationale' in focus_text:
                areas_for_improvement.append('rationale')
            if 'what' in focus_text or 'factual' in focus_text:
                areas_for_improvement.append('factual')
            if 'how' in focus_text or 'procedural' in focus_text:
                areas_for_improvement.append('procedural')

        # If no areas found from focus text, determine based on levels
        if not areas_for_improvement:
            # Find the dimensions with the lowest scores (less than 3)
            dimensions = [
                ('rationale', understanding_levels['rationale_level']),
                ('factual', understanding_levels['factual_level']),
                ('procedural', understanding_levels['procedural_level'])
            ]

            # Sort by level and add any with level < 3
            dimensions.sort(key=lambda x: x[1])
            for dim, level in dimensions:
                if level < 3:
                    areas_for_improvement.append(dim)

        # Print extracted data for debugging
        print(f"User data: {user_data}")
        print(f"Understanding levels: {understanding_levels}")
        print(f"Areas for improvement: {areas_for_improvement}")

        # Check if we successfully parsed the report
        parsed_successfully = (
                len(user_data) > 0 and
                understanding_levels['overall_level'] > 0 and
                any(understanding_levels[key] > 0 for key in ['rationale_level', 'factual_level', 'procedural_level'])
        )

        # Store debug info
        debug_info = {
            "pdf_text_length": len(pdf_text),
            "pdf_text_sample": pdf_text[:500],
            "user_data": user_data,
            "understanding_levels": understanding_levels,
            "areas_for_improvement": areas_for_improvement,
            "parsed_successfully": parsed_successfully
        }
        st.session_state.debug_info = debug_info

        return user_data, understanding_levels, areas_for_improvement, parsed_successfully

    except Exception as e:
        st.error(f"Error processing learning report PDF: {str(e)}")
        import traceback
        st.session_state.pdf_error = traceback.format_exc()
        return {}, {}, [], False


def test_llm_connection():
    """Test the connection to the selected LLM provider"""
    if st.session_state.llm_provider == LLMProvider.LOCAL:
        st.sidebar.success("Local fallback mode active - no API needed.")
        return

    api_key = st.session_state.api_keys.get(st.session_state.llm_provider.value, "")
    if not api_key:
        st.sidebar.error(f"Please enter an API key for {st.session_state.llm_provider.value}")
        return

    try:
        if st.session_state.llm_provider == LLMProvider.GROQ:
            from groq import Groq
            client = Groq(api_key=api_key)
            models = client.models.list()
            st.sidebar.success(f"Successfully connected to Groq! Available models: {len(models.data)}")

        elif st.session_state.llm_provider == LLMProvider.OPENAI:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                models = client.models.list()
                st.sidebar.success(f"Successfully connected to OpenAI! Available models: {len(models.data)}")
            except ImportError:
                st.sidebar.error("OpenAI package not installed. Run: pip install openai")

        elif st.session_state.llm_provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                # Anthropic doesn't have a list models endpoint, so we'll just check if we can make a basic request
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.sidebar.success("Successfully connected to Anthropic!")
            except ImportError:
                st.sidebar.error("Anthropic package not installed. Run: pip install anthropic")

        elif st.session_state.llm_provider == LLMProvider.CUSTOM:
            if not st.session_state.custom_api_url:
                st.sidebar.error("Please enter a Custom API URL")
                return

            # Test with a simple request
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(st.session_state.custom_api_url, headers=headers, timeout=5)
            if response.status_code == 200:
                st.sidebar.success(f"Successfully connected to custom API! Status: {response.status_code}")
            else:
                st.sidebar.error(f"Failed to connect. Status code: {response.status_code}")

    except Exception as e:
        st.sidebar.error(f"Connection test failed: {str(e)}")


def save_assessment_results():
    """Save assessment results for adaptive learning"""
    if 'understanding_report' in st.session_state and st.session_state.understanding_report:
        st.session_state.previous_assessment_results = st.session_state.understanding_report.copy()

        # Generate post-assessment feedback
        if not st.session_state.post_assessment_feedback:
            st.session_state.post_assessment_feedback = generate_post_assessment_feedback(
                st.session_state.understanding_report,
                st.session_state.current_topic
            )


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
                    st.session_state.is_first_login = True  # Set first login flag
                    st.session_state.difficulty_level = "Basic"  # Reset to Basic for new session
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            age = st.number_input("Age", min_value=5, max_value=100, value=25)
            experience = st.text_input("Professional Experience")
            submitted = st.form_submit_button("Register")
            if submitted:
                if save_user_data(new_username, new_password, age, experience):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed")

    # Display dependency warnings if any
    if missing_packages:
        st.warning("⚠️ Some dependencies are missing. The application may not function properly.")
        with st.expander("View Missing Dependencies"):
            st.code("pip install " + " ".join([pkg.split(':')[0] for pkg in missing_packages]))
            for pkg in missing_packages:
                st.write(f"- {pkg}")


def main_application():
    """Main application interface"""
    st.title(f"Welcome {st.session_state.user_data.get('Learning Enthusiast')}!")
    st.sidebar.header("Your Profile")
    st.sidebar.write(f"Age: {st.session_state.user_data.get('age', '')}")
    st.sidebar.write(f"Experience: {st.session_state.user_data.get('experience', '')}")

    # Check for missing dependencies and display warnings in sidebar
    if missing_packages:
        st.sidebar.warning("⚠️ Some dependencies are missing")
        with st.sidebar.expander("Missing Dependencies"):
            st.sidebar.code("pip install " + " ".join([pkg.split(':')[0] for pkg in missing_packages]))
            for pkg in missing_packages:
                st.sidebar.write(f"- {pkg}")

    # Add LLM settings to sidebar
    add_llm_settings()

    # Add adaptive mode toggle in sidebar
    add_adaptive_mode_toggle()

    # Add learning progress summary in sidebar if available
    if 'understanding_report' in st.session_state and st.session_state.understanding_report:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Learning Progress")
        report = st.session_state.understanding_report

        # Display overall level with color
        level = report.get('overall_level', 0)
        level_color = {
            0: "gray",
            1: "red",
            2: "orange",
            3: "blue",
            4: "green",
            5: "purple"
        }.get(level, "gray")

        level_descriptions = {
            0: "Not Assessed",
            1: "Minimal",
            2: "Basic",
            3: "Moderate",
            4: "Good",
            5: "Excellent"
        }

        st.sidebar.markdown(
            f"<div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {level_color}20;'>"
            f"<span style='font-size: 24px; color: {level_color};'>{level}/5</span><br>"
            f"<span style='color: {level_color};'>{level_descriptions[level]}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Show dimension scores
        if all(dim in report for dim in ['rationale_level', 'factual_level', 'procedural_level']):
            st.sidebar.markdown("<small>Dimension Scores:</small>", unsafe_allow_html=True)
            st.sidebar.markdown(
                f"<small>Rationale: {report['rationale_level']}/5 • "
                f"Factual: {report['factual_level']}/5 • "
                f"Procedural: {report['procedural_level']}/5</small>",
                unsafe_allow_html=True
            )

    # If this is first login, show a welcome message with learning instructions
    if st.session_state.is_first_login:
        st.info("""
        ## Welcome to Adaptive Learning!

        This system automatically adjusts to your learning needs:

        1. Start with a topic you want to learn about
        2. Complete the assessment to measure your understanding
        3. The system will analyze your strengths and weaknesses
        4. Future content will be tailored to your learning needs

        You'll start with **Basic** level content and progress based on your performance.
        """)
        st.session_state.is_first_login = False  # Reset the flag

    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["Q&A Mode", "Learning Assessment Mode", "Learning Progress Report"])

    # Store the current tab index in session state if not already present
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

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
                response, conversation_history = get_llm_response(
                    question,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['experience'],
                    conversation_history,
                    client
                )
                st.session_state.chat_history = conversation_history
                st.markdown("### Answer")
                st.write(response)
        if st.session_state.chat_history:
            with st.form(key='followup_form'):
                followup_question = st.text_input("Enter a follow-up question (optional):")
                submit_followup = st.form_submit_button("Submit Follow-up")
                if submit_followup and followup_question:
                    response, conversation_history = get_llm_response(
                        followup_question,
                        st.session_state.user_data['age'],
                        st.session_state.user_data['experience'],
                        st.session_state.chat_history,
                        client
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
                    in_depth_response, _ = get_llm_response(
                        prompt,
                        st.session_state.user_data['age'],
                        st.session_state.user_data['experience'],
                        conversation_history=None,
                        client=client
                    )
                    st.markdown("#### In-depth Explanation")
                    st.write(in_depth_response)

    ############################################
    # Learning Assessment Mode: Learning Topic, Quiz & Direct Questioning
    ############################################
    with tab2:
        st.header("Learning Assessment")

        # Show post-assessment feedback if available
        if st.session_state.post_assessment_feedback and not st.session_state.system_response:
            st.markdown(st.session_state.post_assessment_feedback)
            if st.button("Clear Feedback and Start New Topic"):
                st.session_state.post_assessment_feedback = ""
                st.rerun()

        # Form to set the learning topic or upload PDF
        with st.form(key='learning_form'):
            learning_question = st.text_input("What would you like to learn about?")
            uploaded_file = st.file_uploader("Or upload a PDF to learn from", type="pdf",
                                             help="Upload a PDF document to extract content for learning")

            # Add difficulty level selector
            difficulty_options = ["Basic", "Intermediate", "Advanced"]
            if st.session_state.is_first_login or not st.session_state.adaptive_mode:
                # For first login, force Basic level
                difficulty_level = st.select_slider(
                    "Select content difficulty level:",
                    options=difficulty_options,
                    value="Basic",
                    help="New users start with Basic level content"
                )
            else:
                difficulty_level = st.select_slider(
                    "Select content difficulty level:",
                    options=difficulty_options,
                    value=st.session_state.difficulty_level,
                    help="Choose the difficulty level of content provided"
                )

            # Show adaptive learning status
            if st.session_state.adaptive_mode:
                if 'adaptive_learning_areas' in st.session_state and st.session_state.adaptive_learning_areas:
                    focus_areas = [area.capitalize() for area in st.session_state.adaptive_learning_areas]
                    st.info(
                        f"Adaptive Learning: Content will be tailored to improve your {', '.join(focus_areas)} understanding.")

            submit_learning = st.form_submit_button("Start Learning")

            if submit_learning:
                # Clear post-assessment feedback when starting new learning
                st.session_state.post_assessment_feedback = ""

                # Store the selected difficulty level
                st.session_state.difficulty_level = difficulty_level

                # Check if we should adapt content based on previous results
                adapted_difficulty = difficulty_level
                focus_instruction = ""
                focus_area = None
                focus_level = 0

                if st.session_state.adaptive_mode and 'understanding_report' in st.session_state and st.session_state.understanding_report:
                    adapted_difficulty, focus_instruction, focus_area, focus_level = adapt_content_based_on_assessment(
                        learning_question or uploaded_file.name if uploaded_file else "",
                        st.session_state.understanding_report
                    )

                    # Add feedback message when adaptive mode changes content
                    if adapted_difficulty != difficulty_level or focus_instruction:
                        adjustment_message = f"Based on your previous assessment results, content has been adjusted:"
                        if adapted_difficulty != difficulty_level:
                            adjustment_message += f" Difficulty level changed to {adapted_difficulty}."
                        if focus_instruction:
                            adjustment_message += f" Extra focus on {focus_area} understanding (Level {focus_level}/5)."
                        st.info(adjustment_message)

                if uploaded_file is not None:
                    # Check if PyPDF2 is available before processing
                    if not PYPDF2_AVAILABLE:
                        st.error("PyPDF2 library is not installed. PDF processing is not available.")
                        st.info("To install PyPDF2, run: pip install PyPDF2")
                    else:
                        # Process PDF with adapted difficulty
                        pdf_text, pdf_summary = process_uploaded_pdf(uploaded_file, adapted_difficulty)

                        # Add focus instruction if available
                        if focus_instruction:
                            enhanced_prompt = f"Revise this summary with the following focus: {focus_instruction}\n\n{pdf_summary}"
                            pdf_summary, _ = get_llm_response(
                                enhanced_prompt,
                                st.session_state.user_data['age'],
                                st.session_state.user_data['experience'],
                                conversation_history=None,
                                client=client
                            )

                        st.session_state.pdf_text = pdf_text
                        st.session_state.pdf_summary = pdf_summary
                        st.session_state.system_response = pdf_summary
                        st.session_state.current_topic = uploaded_file.name
                elif learning_question:
                    # Enhanced prompt with adapted difficulty and focus
                    enhanced_prompt = (
                        f"Provide a {adapted_difficulty.lower()}-level educational response on the topic: {learning_question}. "
                        f"If '{adapted_difficulty}' is 'Basic', focus on fundamental concepts with simple explanations. "
                        f"If '{adapted_difficulty}' is 'Intermediate', include more detailed explanations and some technical terms. "
                        f"If '{adapted_difficulty}' is 'Advanced', provide in-depth technical details and sophisticated analysis. "
                    )

                    # Add focus instruction if available
                    if focus_instruction:
                        enhanced_prompt += f"\n\n{focus_instruction}\n\n"

                        # Add section structure based on focus area
                        if focus_area == "rationale":
                            enhanced_prompt += "Structure your response with these sections:\n"
                            enhanced_prompt += "1. Introduction\n"
                            enhanced_prompt += "2. Core Concepts (brief)\n"
                            enhanced_prompt += "3. WHY This Topic Matters (expanded section)\n"
                            enhanced_prompt += "4. Underlying Principles and Reasoning (expanded section)\n"
                            enhanced_prompt += "5. Cause and Effect Relationships (expanded section)\n"
                            enhanced_prompt += "6. Applications (brief)\n"
                        elif focus_area == "factual":
                            enhanced_prompt += "Structure your response with these sections:\n"
                            enhanced_prompt += "1. Introduction\n"
                            enhanced_prompt += "2. Essential Terminology and Definitions (expanded section)\n"
                            enhanced_prompt += "3. Key Facts and Information (expanded section)\n"
                            enhanced_prompt += "4. Important Concepts to Remember (expanded section)\n"
                            enhanced_prompt += "5. Relationships to Other Topics (brief)\n"
                            enhanced_prompt += "6. Applications (brief)\n"
                        elif focus_area == "procedural":
                            enhanced_prompt += "Structure your response with these sections:\n"
                            enhanced_prompt += "1. Introduction\n"
                            enhanced_prompt += "2. Core Concepts (brief)\n"
                            enhanced_prompt += "3. Step-by-Step Application Process (expanded section)\n"
                            enhanced_prompt += "4. Practical Examples (expanded section with multiple examples)\n"
                            enhanced_prompt += "5. Common Implementation Challenges (expanded section)\n"
                            enhanced_prompt += "6. Hands-on Practice Ideas (expanded section)\n"

                    # Complete the prompt
                    enhanced_prompt += f"""
Make sure to adapt the content for a {st.session_state.user_data['age']} year old 
with {st.session_state.user_data['experience']} experience. 
The response should be educational, engaging, and informative.
"""

                    try:
                        system_response, learning_chat_history = get_llm_response(
                            enhanced_prompt,
                            st.session_state.user_data['age'],
                            st.session_state.user_data['experience'],
                            conversation_history=None,
                            client=client
                        )

                        # Check if the response is an error message
                        if "I apologize, but I'm currently experiencing technical difficulties" in system_response or "Error" in system_response:
                            # If it looks like an error response, generate a fallback
                            st.warning("API response error detected. Using fallback content generator.")
                            system_response = generate_fallback_response(learning_question, adapted_difficulty)
                            # Create a basic chat history
                            learning_chat_history = [
                                {"role": "system", "content": "You are a helpful tutor."},
                                {"role": "user", "content": f"Tell me about {learning_question}"},
                                {"role": "assistant", "content": system_response}
                            ]

                        st.session_state.system_response = system_response
                        st.session_state.learning_chat_history = learning_chat_history
                        st.session_state.current_topic = learning_question
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
                        # Use fallback content generator
                        system_response = generate_fallback_response(learning_question, adapted_difficulty)
                        st.session_state.system_response = system_response
                        st.session_state.current_topic = learning_question
                        # Create a basic chat history
                        st.session_state.learning_chat_history = [
                            {"role": "system", "content": "You are a helpful tutor."},
                            {"role": "user", "content": f"Tell me about {learning_question}"},
                            {"role": "assistant", "content": system_response}
                        ]
                else:
                    st.warning("Please either enter a learning topic or upload a PDF file.")
                    st.rerun()

                # Analyze topic complexity to determine appropriate number of questions
                if st.session_state.system_response:
                    st.session_state.topic_complexity = determine_topic_complexity(st.session_state.system_response,
                                                                                   client)

                    # Reset assessment data when a new topic is loaded
                    st.session_state.direct_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.direct_answer_evaluations = []
                    st.success("Learning topic loaded!")

        # Display the system response if available
        if st.session_state.system_response:
            st.markdown("### Review the Topic")

            # If adaptive mode is active and we have focus areas, highlight this at the top
            if (st.session_state.adaptive_mode and
                    'adaptive_learning_areas' in st.session_state and
                    st.session_state.adaptive_learning_areas):
                focus_areas = [area.capitalize() for area in st.session_state.adaptive_learning_areas]
                st.info(f"This content has been customized to improve your {', '.join(focus_areas)} understanding.")

            st.write(st.session_state.system_response)

            # Add follow-up question section for clarification
            st.markdown("### Ask Follow-up Questions")
            st.info("If you need clarification or have questions about the topic, ask below.")

            with st.form(key='learning_followup_form'):
                followup_question = st.text_input("Enter your question about this topic:")
                submit_followup = st.form_submit_button("Get Answer")

                if submit_followup and followup_question:
                    # Get conversation history or create new one if needed
                    conversation_history = st.session_state.learning_chat_history if st.session_state.learning_chat_history else []

                    # Add context about the original topic
                    if len(conversation_history) <= 2:  # If it's a new conversation or has just the first Q&A
                        context_prompt = (
                            f"The user is learning about '{st.session_state.current_topic}'. "
                            f"Please provide a detailed and educational answer to their follow-up question."
                        )
                        conversation_history.append({"role": "system", "content": context_prompt})

                    response, updated_history = get_llm_response(
                        followup_question,
                        st.session_state.user_data['age'],
                        st.session_state.user_data['experience'],
                        conversation_history,
                        client
                    )

                    st.session_state.learning_chat_history = updated_history
                    st.markdown("### Answer")
                    st.write(response)

            # Show previous follow-up questions and answers if they exist
            if len(st.session_state.learning_chat_history) > 2:  # If we have follow-up Q&As
                st.markdown("### Previous Questions")

                # Display only user questions and assistant responses (skip system messages)
                qa_pairs = []
                for i in range(1, len(st.session_state.learning_chat_history), 2):  # Start from user message
                    if i + 1 < len(st.session_state.learning_chat_history):  # Ensure we have a pair
                        user_msg = st.session_state.learning_chat_history[i]["content"]
                        assistant_msg = st.session_state.learning_chat_history[i + 1]["content"]
                        qa_pairs.append((user_msg, assistant_msg))

                # Display the last 3 Q&A pairs (if available)
                for i, (question, answer) in enumerate(qa_pairs[-3:]):
                    with st.expander(f"Q: {question[:50]}..." if len(question) > 50 else f"Q: {question}"):
                        st.write(answer)

            st.markdown("### Assessment Options")
            assessment_type = st.radio(
                "Choose an assessment method:",
                ["Multiple Choice Quiz", "Direct Questions Assessment"]
            )

            # Multiple Choice Quiz Option
            if assessment_type == "Multiple Choice Quiz":
                if st.button("Generate Quiz Questions"):
                    quiz_questions = generate_assessment_questions(st.session_state.system_response, client)
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
                                explanation = get_question_explanation(question_data['question'], correct_answer,
                                                                       client)
                                st.write("Explanation:", explanation)
                        score = (correct_count / len(st.session_state.quiz_questions)) * 100
                        understanding_level = round((score / 100) * 5)  # Convert to 1-5 scale
                        st.write(f"**Your Understanding Level:** {understanding_level}/5")

                        # Save to history (keep score internally but display level)
                        st.session_state.scores.append({
                            'timestamp': datetime.now(),
                            'score': score,
                            'level': understanding_level,
                            'question': "Multiple Choice Quiz",
                            'topic': st.session_state.current_topic
                        })

            # Direct Questions Assessment Option
            elif assessment_type == "Direct Questions Assessment":
                if not st.session_state.direct_questions:
                    # Use the topic complexity to determine number of questions
                    num_questions = 5  # Default
                    question_distribution = None

                    if 'topic_complexity' in st.session_state and st.session_state.topic_complexity:
                        complexity_info = st.session_state.topic_complexity
                        num_questions = complexity_info.get('recommended_questions', 5)
                        question_distribution = complexity_info.get('question_distribution')

                    if st.button("Start Direct Questions Assessment"):
                        direct_questions = generate_direct_questions(
                            st.session_state.system_response,
                            num_questions=num_questions,
                            question_distribution=question_distribution,
                            client=client
                        )
                        st.session_state.direct_questions = direct_questions
                        st.session_state.current_question_index = 0
                        st.session_state.direct_answer_evaluations = []

                # If we have direct questions, display the current one
                if st.session_state.direct_questions:
                    # Display progress
                    total_questions = len(st.session_state.direct_questions)
                    current_index = st.session_state.current_question_index

                    if current_index < total_questions:
                        st.progress((current_index) / total_questions)
                        st.subheader(f"Question {current_index + 1} of {total_questions}")

                        current_question = st.session_state.direct_questions[current_index]
                        question_type = current_question.get('question_type', 'factual')

                        # Display question type badge
                        question_type_color = {
                            'rationale': 'blue',
                            'factual': 'green',
                            'procedural': 'orange'
                        }.get(question_type, 'gray')

                        st.markdown(
                            f"<span style='background-color: {question_type_color}20; "
                            f"color: {question_type_color}; padding: 3px 8px; border-radius: 10px; "
                            f"font-size: 0.8em;'>{question_type.capitalize()} Question</span>",
                            unsafe_allow_html=True
                        )

                        st.markdown(f"**{current_question['question']}**")

                        # Check if the current question has been answered already
                        already_answered = any(eval_item['question_index'] == current_index
                                               for eval_item in st.session_state.direct_answer_evaluations)

                        if not already_answered:
                            with st.form(key=f"direct_question_form_{current_index}"):
                                user_answer = st.text_area("Your answer:", height=150)
                                submitted = st.form_submit_button("Submit Answer")

                                if submitted and user_answer:
                                    # Evaluate the answer
                                    evaluation = evaluate_direct_answer(
                                        {"question": current_question['question'], "answer": user_answer},
                                        current_question['expected_answer'],
                                        question_type,
                                        client
                                    )

                                    # Store the evaluation with question type
                                    st.session_state.direct_answer_evaluations.append({
                                        'question_index': current_index,
                                        'question': current_question['question'],
                                        'question_type': question_type,
                                        'user_answer': user_answer,
                                        'expected_answer': current_question['expected_answer'],
                                        'score': evaluation['score'],
                                        'understanding_level': evaluation.get('understanding_level',
                                                                              round(evaluation['score'] / 20)),
                                        'feedback': evaluation['feedback']
                                    })

                                    # Force rerun to show feedback and navigation
                                    st.rerun()
                        else:
                            # Get the evaluation for this question
                            current_eval = next(eval_item for eval_item in st.session_state.direct_answer_evaluations
                                                if eval_item['question_index'] == current_index)

                            # Display user's answer and feedback
                            st.text_area("Your answer:", value=current_eval['user_answer'], height=150, disabled=True)
                            st.markdown("### Feedback")

                            # Display understanding level if available
                            if 'understanding_level' in current_eval:
                                level = current_eval['understanding_level']
                                level_descriptions = {
                                    1: "Minimal",
                                    2: "Basic",
                                    3: "Moderate",
                                    4: "Good",
                                    5: "Excellent"
                                }
                                st.write(f"**Understanding Level:** {level}/5 - {level_descriptions.get(level, '')}")
                            else:
                                st.write(f"**Score:** {current_eval['score']}/100")

                            st.write(current_eval['feedback'])

                            # Navigation buttons (outside of any form)
                            cols = st.columns([1, 1])
                            if current_index > 0:
                                if cols[0].button("Previous Question"):
                                    st.session_state.current_question_index -= 1
                                    st.rerun()

                            if current_index < total_questions - 1:
                                if cols[1].button("Next Question"):
                                    st.session_state.current_question_index += 1
                                    st.rerun()
                            else:
                                if cols[1].button("View Full Report"):
                                    # Generate report and switch to the report tab
                                    report = generate_understanding_report(st.session_state.direct_answer_evaluations)
                                    st.session_state.understanding_report = report
                                    save_assessment_results()  # Save results for adaptive learning

                                    # Generate post-assessment feedback
                                    st.session_state.post_assessment_feedback = generate_post_assessment_feedback(
                                        report,
                                        st.session_state.current_topic
                                    )

                                    # Clear system response to show feedback on next reload
                                    st.session_state.system_response = ""
                                    st.rerun()

                                st.success("You've completed all the questions!")

                    # If all questions are answered, show summary
                    if current_index >= total_questions - 1 and len(
                            st.session_state.direct_answer_evaluations) == total_questions:
                        st.markdown("### Assessment Summary")

                        # Calculate overall score
                        if st.session_state.direct_answer_evaluations:
                            total_score = sum(
                                eval_item['score'] for eval_item in st.session_state.direct_answer_evaluations)
                            avg_score = total_score / len(st.session_state.direct_answer_evaluations)

                            # Calculate average understanding level
                            avg_level = sum(eval_item.get('understanding_level', round(eval_item['score'] / 20))
                                            for eval_item in st.session_state.direct_answer_evaluations) / len(
                                st.session_state.direct_answer_evaluations)
                            avg_level = round(avg_level)

                            level_descriptions = {
                                1: "Minimal",
                                2: "Basic",
                                3: "Moderate",
                                4: "Good",
                                5: "Excellent"
                            }

                            # Display summary metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Overall Score", f"{avg_score:.1f}/100")
                            with col2:
                                st.metric("Understanding Level",
                                          f"{avg_level}/5 - {level_descriptions.get(avg_level, '')}")

                            # Create a dataframe for the summary
                            summary_df = pd.DataFrame(st.session_state.direct_answer_evaluations)

                            # Add question type if available
                            if 'question_type' in summary_df.columns:
                                summary_df = summary_df[
                                    ['question', 'question_type', 'score', 'understanding_level', 'feedback']]
                            else:
                                summary_df = summary_df[['question', 'score', 'feedback']]

                            st.dataframe(summary_df)

                            # Generate understanding report
                            report = generate_understanding_report(st.session_state.direct_answer_evaluations)
                            st.session_state.understanding_report = report
                            save_assessment_results()  # Save results for adaptive learning

                            # Add to history
                            st.session_state.scores.append({
                                'timestamp': datetime.now(),
                                'score': avg_score,
                                'level': avg_level,
                                'question': "Direct Questions Assessment",
                                'topic': st.session_state.get('current_topic', 'Unnamed Topic')
                            })

                            # Show full report button
                            if st.button("View Full Learning Report"):
                                # Generate post-assessment feedback
                                st.session_state.post_assessment_feedback = generate_post_assessment_feedback(
                                    report,
                                    st.session_state.current_topic
                                )

                                # Clear system response to show feedback on next reload
                                st.session_state.system_response = ""
                                st.rerun()

                            # Reset button (outside of form context)
                            if st.button("Start New Assessment"):
                                st.session_state.direct_questions = []
                                st.session_state.current_question_index = 0
                                st.session_state.direct_answer_evaluations = []
                                st.rerun()
        else:
            # Check if we have post-assessment feedback to display
            if not st.session_state.post_assessment_feedback:
                st.info(
                    "Please click 'Start Learning' by entering a topic above or uploading a PDF to get content for assessment.")

    ############################################
    # Learning Progress Report Tab
    ############################################
    with tab3:
        st.header("📊 Learning Progress Report")

        if ('understanding_report' in st.session_state and
                st.session_state.understanding_report and
                st.session_state.direct_answer_evaluations):

            # Display the learning report
            display_learning_report(
                st.session_state.direct_answer_evaluations,
                st.session_state.scores
            )

        elif st.session_state.scores:
            # If we have scores history but not current assessment, show historical data
            st.subheader("Learning History")

            # Create a dataframe of the scores history
            history_df = pd.DataFrame(st.session_state.scores)

            # Format the timestamps
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df['Date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

            # Display the scores history
            if 'level' in history_df.columns:
                display_df = history_df[['Date', 'question', 'level', 'score']].rename(
                    columns={
                        'question': 'Assessment Type',
                        'level': 'Understanding Level (1-5)',
                        'score': 'Score (0-100)'
                    }
                )
            else:
                display_df = history_df[['Date', 'question', 'score']].rename(
                    columns={
                        'question': 'Assessment Type',
                        'score': 'Score (0-100)'
                    }
                )

            st.dataframe(display_df)

            # Display progress chart
            if len(history_df) > 1:  # Only show if we have multiple entries
                st.subheader("Score Progress Over Time")
                progress_chart_img = create_progress_chart(st.session_state.scores)
                st.image(f"data:image/png;base64,{progress_chart_img}", use_container_width=True)

            st.info("Complete a new assessment to generate a detailed understanding report.")

        else:
            st.info("No assessment data available yet. Complete an assessment to see your learning progress report.")


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
