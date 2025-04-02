import json
import time
import streamlit as st
import importlib.util
import re

# Check if PyPDF2 is available
PYPDF2_AVAILABLE = importlib.util.find_spec('PyPDF2') is not None


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


def get_groq_response(prompt, age, interest, experience, conversation_history=None, client=None, retries=3, delay=2):
    """
    Get enhanced response from Groq with optional conversation chaining.
    This function is maintained for backward compatibility but now calls get_llm_response.
    """
    # Import the function from main to avoid circular imports
    from main import get_llm_response
    return get_llm_response(prompt, age, interest, experience, conversation_history, client, retries, delay)


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
            # Import the function to avoid circular imports
            from main import get_llm_response
            response, _ = get_llm_response(
                prompt,
                st.session_state.user_data['age'],
                st.session_state.user_data['interest'],
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

    # Import the function to avoid circular imports
    from main import get_llm_response

    response, _ = get_llm_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
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

    # Import the function to avoid circular imports
    from main import get_llm_response

    response, _ = get_llm_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
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
        # Import the function to avoid circular imports
        from main import get_llm_response

        response, _ = get_llm_response(
            prompt,
            st.session_state.user_data['age'],
            st.session_state.user_data['interest'],
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


def get_question_explanation(question, correct_answer, client=None):
    """
    Get an in-depth explanation from the LLM as to why the given correct answer
    is correct for the provided question.
    """
    prompt = (
        f"Provide an in-depth explanation for the following question:\n\n"
        f"Question: {question}\n\n"
        f"Explain why the correct answer is '{correct_answer}' and highlight the key points that support it."
    )

    # Import the function to avoid circular imports
    from main import get_llm_response

    explanation, _ = get_llm_response(
        prompt,
        st.session_state.user_data['age'],
        st.session_state.user_data['interest'],
        st.session_state.user_data['experience'],
        conversation_history=None,
        client=client
    )
    return explanation


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.

    Args:
        pdf_file: A file-like object containing PDF data

    Returns:
        str: Extracted text from the PDF
    """
    # Check if PyPDF2 is available
    if not PYPDF2_AVAILABLE:
        st.error("PyPDF2 library is not installed. PDF processing is not available.")
        st.info("To install PyPDF2, run: pip install PyPDF2")
        return "Error: PyPDF2 library is not installed. PDF processing is not available."

    try:
        import PyPDF2
        import io

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages
        num_pages = len(pdf_reader.pages)

        # Extract text from each page
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"

        return text
    except Exception as e:
        error_message = f"Error extracting text from PDF: {str(e)}"
        st.error(error_message)
        return error_message