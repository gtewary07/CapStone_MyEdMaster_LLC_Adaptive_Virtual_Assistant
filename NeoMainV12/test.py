import streamlit as st
import time
import json
import hashlib
import re
import requests
import base64
import io
from enum import Enum
from datetime import datetime


# Define LLM API options as an enum
class LLMProvider(Enum):
    GROQ = "Groq"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    CUSTOM = "Custom API"
    LOCAL = "Local Fallback (No API)"


# Check if reportlab is available for PDF generation
def check_reportlab():
    """Check if reportlab is installed"""
    try:
        import reportlab
        return True
    except ImportError:
        return False


# Initialize session state variables
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'system_response' not in st.session_state:
        st.session_state.system_response = ""
    if 'learning_chat_history' not in st.session_state:
        st.session_state.learning_chat_history = []
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = LLMProvider.LOCAL  # Default to local fallback for demo
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            LLMProvider.GROQ.value: "",
            LLMProvider.OPENAI.value: "",
            LLMProvider.ANTHROPIC.value: "",
            LLMProvider.CUSTOM.value: ""
        }
    if 'custom_api_url' not in st.session_state:
        st.session_state.custom_api_url = ""
    if 'understanding_level' not in st.session_state:
        st.session_state.understanding_level = 3  # Default to middle level
    if 'dimension_levels' not in st.session_state:
        st.session_state.dimension_levels = {
            'rationale': 3,
            'factual': 3,
            'procedural': 3
        }
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'pdf_download_ready' not in st.session_state:
        st.session_state.pdf_download_ready = False
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None


# Simple user verification for the demo
def verify_user(username, password):
    """Verify user credentials (hardcoded for demo)"""
    # For demo purposes only - in a real app, use secure authentication
    if username == "demo" and password == "password":
        return {
            'username': 'demo',
            'age': 20,
            'experience': 'Undergraduate student in Mathematics'
        }
    return None


# Generate a fallback response based on understanding level
def generate_enhanced_fallback_response(topic, understanding_level=3, dimension_levels=None):
    """
    Generate a fallback response adjusted for understanding level and dimension focus.
    """
    if dimension_levels is None:
        dimension_levels = {
            'rationale': understanding_level,
            'factual': understanding_level,
            'procedural': understanding_level
        }

    # Find the dimension with the lowest level (area to focus on)
    focus_dimension = min(dimension_levels, key=dimension_levels.get)
    min_level = dimension_levels[focus_dimension]

    # Determine overall complexity based on understanding level
    if understanding_level <= 2:
        complexity = "basic"
        language = "simple terms with minimal technical vocabulary"
    elif understanding_level == 3:
        complexity = "intermediate"
        language = "balanced vocabulary with important technical terms explained"
    else:  # understanding_level >= 4
        complexity = "advanced"
        language = "technical vocabulary appropriate for the field"

    # Determine structure based on focus dimension
    if focus_dimension == 'rationale' and min_level < 4:
        structure = """
## Why This Topic Matters
This section would explore the reasoning behind key concepts and why they are important.

## Underlying Principles
This section would explain the fundamental principles that govern this topic.

## Cause and Effect Relationships
This section would explore how different factors interact and influence outcomes.
"""
    elif focus_dimension == 'factual' and min_level < 4:
        structure = """
## Essential Terminology
This section would define all key terms and concepts clearly.

## Key Facts and Information
This section would provide the most important factual information about the topic.

## Important Concepts to Remember
This section would highlight the most critical concepts that should be memorized.
"""
    elif focus_dimension == 'procedural' and min_level < 4:
        structure = """
## Step-by-Step Application
This section would provide a clear sequence of steps to apply this knowledge.

## Practical Examples
This section would demonstrate how to use this knowledge in real situations.

## Common Implementation Challenges
This section would address typical problems and how to overcome them.
"""
    else:
        structure = """
## Key Concepts
This section would cover the most important aspects of the topic.

## Applications
This section would explore how this knowledge is applied in practice.

## Advanced Considerations
This section would discuss more complex aspects for deeper understanding.
"""

    # Create the response
    return f"""
# Understanding {topic} (Level {understanding_level}/5)

This is a {complexity}-level overview of {topic}, tailored for your current understanding level.
The content uses {language} and focuses particularly on {'reasoning and causes' if focus_dimension == 'rationale' else 'key facts and concepts' if focus_dimension == 'factual' else 'practical application and implementation'}.

{structure}

## Understanding Level Breakdown
- Rationale Understanding: {dimension_levels['rationale']}/5 - {'Needs improvement' if dimension_levels['rationale'] < 3 else 'Good understanding' if dimension_levels['rationale'] < 5 else 'Excellent understanding'}
- Factual Understanding: {dimension_levels['factual']}/5 - {'Needs improvement' if dimension_levels['factual'] < 3 else 'Good understanding' if dimension_levels['factual'] < 5 else 'Excellent understanding'}
- Procedural Understanding: {dimension_levels['procedural']}/5 - {'Needs improvement' if dimension_levels['procedural'] < 3 else 'Good understanding' if dimension_levels['procedural'] < 5 else 'Excellent understanding'}

*Note: This is a sample response demonstrating how content would be tailored to your understanding level.*
"""


# Get LLM client based on provider selection
def get_llm_client():
    """Get appropriate client for selected LLM provider"""
    provider = st.session_state.llm_provider

    if provider == LLMProvider.GROQ:
        try:
            from groq import Groq
            api_key = st.session_state.api_keys.get(LLMProvider.GROQ.value)
            if not api_key:
                st.warning("No API key provided for Groq. Using local fallback.")
                return None
            return Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            return None

    elif provider == LLMProvider.OPENAI:
        try:
            import openai
            api_key = st.session_state.api_keys.get(LLMProvider.OPENAI.value)
            if not api_key:
                st.warning("No API key provided for OpenAI. Using local fallback.")
                return None
            return openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None

    elif provider == LLMProvider.ANTHROPIC:
        try:
            import anthropic
            api_key = st.session_state.api_keys.get(LLMProvider.ANTHROPIC.value)
            if not api_key:
                st.warning("No API key provided for Anthropic. Using local fallback.")
                return None
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Anthropic client: {str(e)}")
            return None

    elif provider == LLMProvider.CUSTOM:
        api_key = st.session_state.api_keys.get(LLMProvider.CUSTOM.value)
        url = st.session_state.custom_api_url
        if not api_key or not url:
            st.warning("API key or URL not provided for Custom API. Using local fallback.")
            return None
        return {
            "type": "custom",
            "api_key": api_key,
            "url": url
        }

    else:  # LOCAL
        return None


# Enhanced LLM response function that considers understanding level
def enhanced_get_llm_response(prompt, age, experience, conversation_history=None, client=None, understanding_level=3,
                              dimension_levels=None, retries=3, delay=2):
    """
    Get response from LLM with understanding level and dimension focus.

    Parameters:
    - prompt: User's query
    - age: User's age
    - experience: User's professional background
    - conversation_history: Previous messages
    - client: LLM API client
    - understanding_level: Overall understanding level (1-5)
    - dimension_levels: Dict with levels for rationale, factual, procedural dimensions
    - retries: Number of retry attempts
    - delay: Delay between retries
    """
    if dimension_levels is None:
        dimension_levels = {
            'rationale': understanding_level,
            'factual': understanding_level,
            'procedural': understanding_level
        }

    # Find dimension with lowest level to focus on
    focus_dimension = min(dimension_levels, key=dimension_levels.get)
    min_level = dimension_levels[focus_dimension]

    if conversation_history is None:
        # Enhanced system prompt with understanding level and dimension focus
        system_prompt = (
            f"You are an intelligent, patient, and engaging tutor for a {age} year old person "
            f"with {experience} experience. The learner has the following understanding levels:\n"
            f"- Overall Understanding: {understanding_level}/5\n"
            f"- Rationale Understanding (why things work): {dimension_levels['rationale']}/5\n"
            f"- Factual Understanding (key information): {dimension_levels['factual']}/5\n"
            f"- Procedural Understanding (how to apply): {dimension_levels['procedural']}/5\n\n"
            f"Your responses should be:"
        )

        # Add general guidelines for all responses
        system_prompt += (
            f"\n1. Clear and age-appropriate, using analogies and examples suited to the user's background"
            f"\n2. Well-structured with headings and sections for complex topics"
            f"\n3. Engaging, with questions to check understanding where appropriate"
            f"\n4. Comprehensive but not overwhelming, with a balance of depth and accessibility"
            f"\n5. Connected to the user's professional background ({experience})"
            f"\n6. Educational in tone, focusing on building understanding"
        )

        # Add specifics based on understanding level
        if understanding_level <= 2:
            system_prompt += (
                f"\n\nSince the overall understanding level is basic ({understanding_level}/5):"
                f"\n- Use simple language and clear explanations with minimal technical terms"
                f"\n- Break down complex concepts into smaller, more manageable pieces"
                f"\n- Use many concrete examples and visual analogies related to everyday experiences"
                f"\n- Emphasize foundational knowledge and core principles before details"
            )
        elif understanding_level == 3:
            system_prompt += (
                f"\n\nSince the overall understanding level is intermediate ({understanding_level}/5):"
                f"\n- Balance fundamental concepts with more detailed explanations"
                f"\n- Introduce field-specific terminology with clear definitions"
                f"\n- Use a mix of basic and more nuanced examples"
                f"\n- Begin connecting concepts to broader principles"
            )
        else:  # understanding_level >= 4
            system_prompt += (
                f"\n\nSince the overall understanding level is advanced ({understanding_level}/5):"
                f"\n- Use more sophisticated analysis and nuanced explanations"
                f"\n- Include technical details and domain-specific terminology"
                f"\n- Discuss theoretical frameworks and underlying principles in depth"
                f"\n- Reference advanced connections between concepts"
            )

        # Add focus area based on dimension with lowest score
        if focus_dimension == 'rationale' and min_level < 4:
            system_prompt += (
                f"\n\nPut extra emphasis on explaining the 'why' behind concepts since the rationale "
                f"understanding level is {min_level}/5:"
                f"\n- Focus on reasons, causes, and implications"
                f"\n- Explain underlying principles and reasoning"
                f"\n- Connect causes to effects"
                f"\n- Provide context for why concepts are important"
            )
        elif focus_dimension == 'factual' and min_level < 4:
            system_prompt += (
                f"\n\nPut extra emphasis on fundamental facts and definitions since the factual "
                f"understanding level is {min_level}/5:"
                f"\n- Include more foundational facts, definitions, and key terms"
                f"\n- Focus on building a strong factual base"
                f"\n- Provide clear, memorable definitions"
                f"\n- Organize facts in a logical structure"
            )
        elif focus_dimension == 'procedural' and min_level < 4:
            system_prompt += (
                f"\n\nPut extra emphasis on practical application since the procedural "
                f"understanding level is {min_level}/5:"
                f"\n- Include more step-by-step procedures and examples"
                f"\n- Focus on how to apply concepts"
                f"\n- Provide clear, actionable instructions"
                f"\n- Include scenarios for practical implementation"
            )

        conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

    conversation_history.append({"role": "user", "content": prompt})

    # Check if we're in local fallback mode
    if st.session_state.llm_provider == LLMProvider.LOCAL:
        # Generate a simple response based on the prompt
        response = generate_enhanced_fallback_response(
            prompt.split(":")[-1] if ":" in prompt else prompt,
            understanding_level,
            dimension_levels
        )
        conversation_history.append({"role": "assistant", "content": response})
        return response, conversation_history

    # If client is None, get a new client
    if client is None:
        client = get_llm_client()
        if client is None:
            # If we can't get a client, use fallback
            response = generate_enhanced_fallback_response(
                prompt.split(":")[-1] if ":" in prompt else prompt,
                understanding_level,
                dimension_levels
            )
            conversation_history.append({"role": "assistant", "content": response})
            return response, conversation_history

    # Try to get a response from the API
    try:
        provider = st.session_state.llm_provider

        if provider == LLMProvider.GROQ:
            chat_completion = client.chat.completions.create(
                messages=conversation_history,
                model="llama-3.1-8b-instant",  # Start with a fast model
                temperature=0.7,
                max_tokens=2048
            )
            assistant_message = chat_completion.choices[0].message.content

        elif provider == LLMProvider.OPENAI:
            chat_completion = client.chat.completions.create(
                messages=conversation_history,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=2048
            )
            assistant_message = chat_completion.choices[0].message.content

        elif provider == LLMProvider.ANTHROPIC:
            # Convert for Anthropic format
            messages = []
            system = ""
            for msg in conversation_history:
                if msg["role"] == "system":
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

        elif provider == LLMProvider.CUSTOM:
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
                assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise Exception(f"API returned status code {response.status_code}")

        conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, conversation_history

    except Exception as e:
        st.error(f"Error getting response from API: {str(e)}")

        # Use fallback response
        response = generate_enhanced_fallback_response(
            prompt.split(":")[-1] if ":" in prompt else prompt,
            understanding_level,
            dimension_levels
        )
        conversation_history.append({"role": "assistant", "content": response})
        return response, conversation_history


# Generate enhanced learning content
def generate_enhanced_learning_content(topic, age, experience, understanding_level=3, dimension_levels=None,
                                       client=None):
    """
    Generate learning content tailored to understanding level and dimension focus.

    Parameters:
    - topic: Learning topic
    - age: User's age
    - experience: User's professional background
    - understanding_level: Overall understanding level (1-5)
    - dimension_levels: Dict with levels for rationale, factual, procedural dimensions
    - client: LLM API client
    """
    if dimension_levels is None:
        dimension_levels = {
            'rationale': understanding_level,
            'factual': understanding_level,
            'procedural': understanding_level
        }

    # Find dimension with lowest level to focus on
    focus_dimension = min(dimension_levels, key=dimension_levels.get)
    min_level = dimension_levels[focus_dimension]

    # Create prompt based on understanding levels
    prompt = f"Provide an educational response on the topic: {topic}."

    # Add focus instruction based on lowest dimension
    if focus_dimension == 'rationale' and min_level < 4:
        prompt += (
            f"\n\nPut extra emphasis on explaining the 'why' behind concepts since the learner's "
            f"rationale understanding level is {min_level}/5. Focus on reasons, causes, and implications."
            f"\n\nStructure your response with these sections:"
            f"\n1. Introduction"
            f"\n2. Core Concepts (brief)"
            f"\n3. WHY This Topic Matters (expanded section)"
            f"\n4. Underlying Principles and Reasoning (expanded section)"
            f"\n5. Cause and Effect Relationships (expanded section)"
            f"\n6. Applications (brief)"
        )
    elif focus_dimension == 'factual' and min_level < 4:
        prompt += (
            f"\n\nPut extra emphasis on fundamental facts and definitions since the learner's "
            f"factual understanding level is {min_level}/5. Focus on building a strong knowledge base."
            f"\n\nStructure your response with these sections:"
            f"\n1. Introduction"
            f"\n2. Essential Terminology and Definitions (expanded section)"
            f"\n3. Key Facts and Information (expanded section)"
            f"\n4. Important Concepts to Remember (expanded section)"
            f"\n5. Relationships to Other Topics (brief)"
            f"\n6. Applications (brief)"
        )
    elif focus_dimension == 'procedural' and min_level < 4:
        prompt += (
            f"\n\nPut extra emphasis on practical application since the learner's "
            f"procedural understanding level is {min_level}/5. Focus on how to apply concepts."
            f"\n\nStructure your response with these sections:"
            f"\n1. Introduction"
            f"\n2. Core Concepts (brief)"
            f"\n3. Step-by-Step Application Process (expanded section)"
            f"\n4. Practical Examples (expanded section with multiple examples)"
            f"\n5. Common Implementation Challenges (expanded section)"
            f"\n6. Hands-on Practice Ideas (expanded section)"
        )

    # Add overall level adjustment
    if understanding_level <= 2:
        prompt += (
            f"\n\nThe learner has a basic understanding level ({understanding_level}/5), so:"
            f"\n- Use simple language with minimal technical terms"
            f"\n- Focus on foundational concepts"
            f"\n- Provide many concrete examples"
            f"\n- Break complex ideas into smaller pieces"
        )
    elif understanding_level == 3:
        prompt += (
            f"\n\nThe learner has an intermediate understanding level ({understanding_level}/5), so:"
            f"\n- Balance basic and advanced concepts"
            f"\n- Introduce technical terms with clear definitions"
            f"\n- Provide some theoretical background"
            f"\n- Connect concepts to broader principles"
        )
    else:  # understanding_level >= 4
        prompt += (
            f"\n\nThe learner has an advanced understanding level ({understanding_level}/5), so:"
            f"\n- Include technical details and domain-specific terminology"
            f"\n- Explore theoretical frameworks in depth"
            f"\n- Discuss nuances and advanced connections"
            f"\n- Include recent developments in the field"
        )

    # Complete the prompt
    prompt += f"\n\nMake sure to adapt the content for a {age} year old with {experience} experience."

    # Get response using the enhanced function
    try:
        response, conversation_history = enhanced_get_llm_response(
            prompt,
            age,
            experience,
            conversation_history=None,
            client=client,
            understanding_level=understanding_level,
            dimension_levels=dimension_levels
        )

        return response, conversation_history
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        # Use fallback response
        fallback_response = generate_enhanced_fallback_response(
            topic,
            understanding_level,
            dimension_levels
        )

        # Create basic conversation history
        conversation_history = [
            {"role": "system", "content": f"You are a helpful tutor."},
            {"role": "user", "content": f"Tell me about {topic}"},
            {"role": "assistant", "content": fallback_response}
        ]

        return fallback_response, conversation_history


# Function to generate a PDF report
def generate_pdf_report(topic, content, user_data, understanding_level, dimension_levels):
    """
    Generate a PDF report of the learning content with proper Markdown formatting.

    Parameters:
    - topic: Learning topic
    - content: Generated content
    - user_data: User information
    - understanding_level: Overall understanding level
    - dimension_levels: Dict with levels for each dimension

    Returns:
    - PDF file as bytes
    """
    # Check if reportlab is available
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListItem, ListFlowable
        from reportlab.lib.units import inch
    except ImportError:
        return None

    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        subheading_style = styles['Heading2']
        normal_style = styles['Normal']

        # Create a bold style
        bold_style = ParagraphStyle(
            'BoldStyle',
            parent=normal_style,
            fontName='Helvetica-Bold'
        )

        # Create a bullet style
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=normal_style,
            leftIndent=20,
            firstLineIndent=0
        )

        # Create a custom style for info box
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=normal_style,
            backColor=colors.lightblue,
            borderColor=colors.blue,
            borderWidth=1,
            borderPadding=5,
            borderRadius=5,
            spaceBefore=10,
            spaceAfter=10
        )

        # Build the PDF content
        elements = []

        # Add title
        elements.append(Paragraph(f"Learning Report: {topic}", title_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Add generation date
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {current_time}", normal_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Add user information section
        elements.append(Paragraph("Learner Profile", heading_style))

        # Create a table for user info
        user_data_list = [
            ["User", user_data.get('username', 'N/A')],
            ["Age", str(user_data.get('age', 'N/A'))],
            ["Experience", user_data.get('experience', 'N/A')]
        ]

        user_table = Table(user_data_list, colWidths=[1.5 * inch, 4 * inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))

        elements.append(user_table)
        elements.append(Spacer(1, 0.25 * inch))

        # Add understanding level section
        elements.append(Paragraph("Understanding Levels", heading_style))

        # Add overall understanding level
        level_descriptions = {
            1: "Minimal",
            2: "Basic",
            3: "Moderate",
            4: "Good",
            5: "Excellent"
        }

        elements.append(Paragraph(
            f"Overall Understanding: Level {understanding_level}/5 ({level_descriptions.get(understanding_level, 'Unknown')})",
            subheading_style
        ))

        # Create a table for dimension levels
        dimension_data = [
            ["Dimension", "Level", "Description"],
            ["Rationale (Why)", str(dimension_levels['rationale']) + "/5",
             level_descriptions.get(dimension_levels['rationale'], 'Unknown')],
            ["Factual (What)", str(dimension_levels['factual']) + "/5",
             level_descriptions.get(dimension_levels['factual'], 'Unknown')],
            ["Procedural (How)", str(dimension_levels['procedural']) + "/5",
             level_descriptions.get(dimension_levels['procedural'], 'Unknown')]
        ]

        dim_table = Table(dimension_data, colWidths=[2 * inch, 1 * inch, 2.5 * inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))

        elements.append(dim_table)
        elements.append(Spacer(1, 0.25 * inch))

        # Find focus dimension (lowest level)
        focus_dimension = min(dimension_levels, key=dimension_levels.get)
        focus_level = dimension_levels[focus_dimension]

        # Add focus area info box
        focus_text = (
            f"<b>Learning Focus:</b> This content emphasizes "
            f"{'understanding WHY (reasoning and causes)' if focus_dimension == 'rationale' else 'WHAT (key facts and definitions)' if focus_dimension == 'factual' else 'HOW (practical application)'}"
            f" since your {focus_dimension} understanding is at level {focus_level}/5."
        )

        elements.append(Paragraph(focus_text, info_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Add content section
        elements.append(Paragraph("Learning Content", heading_style))

        # Process Markdown-like content into ReportLab paragraphs
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Handle headings
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], heading_style))
                i += 1
                continue

            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], subheading_style))
                i += 1
                continue

            # Handle bullet points
            elif line.startswith('* '):
                # Collect all bullet points in a sequence
                bullet_items = []

                while i < len(lines) and lines[i].strip().startswith('* '):
                    # Get the bullet point text without the *
                    bullet_text = lines[i].strip()[2:]

                    # Process bold formatting in the bullet point
                    bullet_text = process_bold_text(bullet_text)

                    bullet_items.append(Paragraph(bullet_text, bullet_style))
                    i += 1

                # Create a bullet list with all collected items
                bullet_list = ListFlowable(
                    bullet_items,
                    bulletType='bullet',
                    leftIndent=20,
                    start=None
                )

                elements.append(bullet_list)
                continue

            # Handle empty lines
            elif not line:
                i += 1
                continue

            # Handle regular paragraphs
            else:
                # Collect all lines in the paragraph
                paragraph_lines = []

                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('# ', '## ', '* ')):
                    paragraph_lines.append(lines[i])
                    i += 1

                if paragraph_lines:
                    # Join the paragraph lines and process bold formatting
                    paragraph_text = ' '.join(paragraph_lines)
                    paragraph_text = process_bold_text(paragraph_text)

                    elements.append(Paragraph(paragraph_text, normal_style))
                    elements.append(Spacer(1, 0.1 * inch))

        # Add footer
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(
            "This report was generated by the Enhanced Learning System based on your understanding level profile.",
            normal_style
        ))

        # Build the PDF
        doc.build(elements)

        # Get the value from the buffer
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    except Exception as e:
        return None


def process_bold_text(text):
    """
    Process Markdown bold formatting (** **) and convert it to ReportLab bold formatting.

    Parameters:
    - text: Text that may contain Markdown bold formatting

    Returns:
    - Text with ReportLab bold formatting
    """
    # Find all occurrences of **text**
    import re

    # Replace **text** with <b>text</b> for ReportLab
    processed_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    return processed_text


# Function to create a download link for a PDF
def get_pdf_download_link(pdf_data, filename="learning_report.pdf"):
    """
    Generates a download link for the PDF data.

    Parameters:
    - pdf_data: The PDF data as bytes
    - filename: The filename to use for the download

    Returns:
    - HTML string with download link
    """
    if pdf_data is None:
        return ""

    # Encode the PDF data as base64
    b64 = base64.b64encode(pdf_data).decode()

    # Create a download link
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'

    return href


# Add LLM settings to sidebar
def add_llm_settings():
    """Add LLM API settings to the sidebar"""
    with st.sidebar.expander("🤖 LLM API Settings", expanded=False):
        # LLM Provider Selection
        provider_options = [provider.value for provider in LLMProvider]
        current_provider = st.session_state.llm_provider.value

        selected_provider = st.radio(
            "Select LLM Provider:",
            provider_options,
            index=provider_options.index(current_provider)
        )

        # Update provider in session state
        st.session_state.llm_provider = next(
            provider for provider in LLMProvider if provider.value == selected_provider)

        # Show API key input if not using local fallback
        if st.session_state.llm_provider != LLMProvider.LOCAL:
            api_key = st.text_input(
                f"{st.session_state.llm_provider.value} API Key:",
                value=st.session_state.api_keys.get(st.session_state.llm_provider.value, ""),
                type="password"
            )
            st.session_state.api_keys[st.session_state.llm_provider.value] = api_key

            # Show custom API URL for custom provider
            if st.session_state.llm_provider == LLMProvider.CUSTOM:
                custom_url = st.text_input(
                    "Custom API URL:",
                    value=st.session_state.custom_api_url
                )
                st.session_state.custom_api_url = custom_url

        if st.button("Test Connection"):
            test_llm_connection()


# Test LLM connection
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
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                models = client.models.list()
                st.sidebar.success(f"Successfully connected to Groq!")
            except Exception as e:
                st.sidebar.error(f"Groq connection failed: {str(e)}")

        elif st.session_state.llm_provider == LLMProvider.OPENAI:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                models = client.models.list()
                st.sidebar.success(f"Successfully connected to OpenAI!")
            except Exception as e:
                st.sidebar.error(f"OpenAI connection failed: {str(e)}")

        elif st.session_state.llm_provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                st.sidebar.success("Successfully connected to Anthropic!")
            except Exception as e:
                st.sidebar.error(f"Anthropic connection failed: {str(e)}")

        elif st.session_state.llm_provider == LLMProvider.CUSTOM:
            if not st.session_state.custom_api_url:
                st.sidebar.error("Please enter a Custom API URL")
                return

            # Test with a simple request
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(st.session_state.custom_api_url, headers=headers, timeout=5)
            if response.status_code == 200:
                st.sidebar.success(f"Successfully connected to custom API!")
            else:
                st.sidebar.error(f"Failed to connect. Status code: {response.status_code}")

    except Exception as e:
        st.sidebar.error(f"Connection test failed: {str(e)}")


# Render understanding level with visual indicator
def render_understanding_level(level, label=None):
    """Render visual indicator of understanding level"""
    level_colors = {
        1: "#FF5733",  # Red
        2: "#FFC300",  # Yellow
        3: "#3498DB",  # Blue
        4: "#2ECC71",  # Green
        5: "#9B59B6"  # Purple
    }

    level_labels = {
        1: "Minimal",
        2: "Basic",
        3: "Moderate",
        4: "Good",
        5: "Excellent"
    }

    level = max(1, min(5, level))  # Ensure level is between 1-5
    level_color = level_colors.get(level, "#CCCCCC")
    level_text = level_labels.get(level, "Unknown")

    # Create visual indicator
    html = f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; align-items: center;">
            <div style="
                width: 30px; 
                height: 30px; 
                border-radius: 50%; 
                background-color: {level_color}; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                color: white; 
                font-weight: bold;
                margin-right: 10px;
            ">{level}</div>
            <div>
                <div style="font-weight: bold;">{level_text} Understanding</div>
                {f'<div style="font-size: 0.8em; color: #666;">{label}</div>' if label else ''}
            </div>
        </div>
    </div>
    """

    return html


# Login page
def login_page():
    """Display login page"""
    st.header("Login")

    with st.form("login_form"):
        username = st.text_input("Username (use 'demo')")
        password = st.text_input("Password (use 'password')", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user_data = verify_user(username, password)
            if user_data:
                st.session_state.logged_in = True
                st.session_state.user_data = user_data
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try demo/password")

    # Show demo credentials
    st.info("Demo Credentials:\nUsername: demo\nPassword: password")

    # Show PDF dependencies check
    if not check_reportlab():
        st.warning(
            "ReportLab is not installed. PDF reports will not be available. "
            "To install ReportLab, run: pip install reportlab"
        )


# Main application interface
def main_application():
    """Main application interface"""
    st.sidebar.header("Your Profile")
    st.sidebar.write(f"Age: {st.session_state.user_data.get('age', '')}")
    st.sidebar.write(f"Experience: {st.session_state.user_data.get('experience', '')}")

    # Add LLM settings to sidebar
    add_llm_settings()

    # Understanding level settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Understanding Level Settings")

    # Overall understanding level slider
    overall_level = st.sidebar.slider(
        "Overall Understanding Level",
        min_value=1,
        max_value=5,
        value=st.session_state.understanding_level,
        help="Set overall understanding level (1=Minimal, 5=Excellent)"
    )
    st.session_state.understanding_level = overall_level

    # Individual dimension level sliders
    st.sidebar.markdown("#### Dimension Levels")

    rationale_level = st.sidebar.slider(
        "Rationale Understanding (Why)",
        min_value=1,
        max_value=5,
        value=st.session_state.dimension_levels.get('rationale', overall_level),
        help="Understanding of reasons and causes"
    )

    factual_level = st.sidebar.slider(
        "Factual Understanding (What)",
        min_value=1,
        max_value=5,
        value=st.session_state.dimension_levels.get('factual', overall_level),
        help="Knowledge of key facts and concepts"
    )

    procedural_level = st.sidebar.slider(
        "Procedural Understanding (How)",
        min_value=1,
        max_value=5,
        value=st.session_state.dimension_levels.get('procedural', overall_level),
        help="Ability to apply knowledge practically"
    )

    # Update dimension levels in session state
    st.session_state.dimension_levels = {
        'rationale': rationale_level,
        'factual': factual_level,
        'procedural': procedural_level
    }

    # Main content area

    # Display understanding levels
    st.subheader("Current Understanding Levels")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            render_understanding_level(overall_level, "Overall"),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            render_understanding_level(rationale_level, "Rationale (Why)"),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            render_understanding_level(factual_level, "Factual (What)"),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            render_understanding_level(procedural_level, "Procedural (How)"),
            unsafe_allow_html=True
        )

    # Understanding level explanation
    with st.expander("Understanding Level Explanation"):
        st.markdown("""
        ### Understanding Levels Explained

        #### Overall Understanding Level
        Sets the general complexity of content:
        - **Level 1-2**: Basic with simple language and fundamental concepts
        - **Level 3**: Intermediate with balanced approach
        - **Level 4-5**: Advanced with technical details and sophisticated analysis

        #### Dimension Levels

        **Rationale Understanding (Why)**  
        Focuses on reasoning, causes, and implications:
        - Low level: System explains WHY concepts matter with extra emphasis
        - High level: System assumes you understand underlying reasons

        **Factual Understanding (What)**  
        Focuses on key facts, definitions, and information:
        - Low level: System emphasizes foundational knowledge with clear definitions
        - High level: System assumes you know the basics

        **Procedural Understanding (How)**  
        Focuses on application and implementation:
        - Low level: System provides step-by-step instructions and examples
        - High level: System assumes you know how to apply concepts

        The system automatically focuses on your weakest dimension (lowest score).
        """)

    # Topic input form
    with st.form("topic_form"):
        learning_topic = st.text_input(
            "What would you like to learn about?",
            value=st.session_state.current_topic
        )

        submitted = st.form_submit_button("Generate Content")

        if submitted and learning_topic:
            st.session_state.current_topic = learning_topic

            # Get client
            client = get_llm_client()

            # Generate content
            with st.spinner("Generating learning content..."):
                response, chat_history = generate_enhanced_learning_content(
                    learning_topic,
                    st.session_state.user_data['age'],
                    st.session_state.user_data['experience'],
                    st.session_state.understanding_level,
                    st.session_state.dimension_levels,
                    client
                )

                st.session_state.system_response = response
                st.session_state.learning_chat_history = chat_history

                # Reset PDF download data
                st.session_state.pdf_download_ready = False
                st.session_state.pdf_data = None

    # Display generated content
    if st.session_state.system_response:
        st.markdown("### Generated Learning Content")

        # Show understanding level context
        focus_dimension = min(st.session_state.dimension_levels, key=st.session_state.dimension_levels.get)
        min_level = st.session_state.dimension_levels[focus_dimension]

        st.info(
            f"This content is tailored for an overall understanding level of {st.session_state.understanding_level}/5 "
            f"with specific focus on {'rationale (why)' if focus_dimension == 'rationale' else 'factual (what)' if focus_dimension == 'factual' else 'procedural (how)'} "
            f"understanding (level {min_level}/5)."
        )

        # Display content
        st.markdown(st.session_state.system_response)

        # Add PDF download option
        if check_reportlab():
            # Check if PDF data is already generated
            if not st.session_state.pdf_download_ready:
                # Generate PDF report
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        pdf_data = generate_pdf_report(
                            st.session_state.current_topic,
                            st.session_state.system_response,
                            st.session_state.user_data,
                            st.session_state.understanding_level,
                            st.session_state.dimension_levels
                        )

                        if pdf_data:
                            st.session_state.pdf_data = pdf_data
                            st.session_state.pdf_download_ready = True
                            st.success("PDF report generated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to generate PDF report.")
            else:
                # Show download link for previously generated PDF
                st.markdown(
                    get_pdf_download_link(
                        st.session_state.pdf_data,
                        f"{st.session_state.current_topic}_learning_report.pdf".replace(" ", "_")
                    ),
                    unsafe_allow_html=True
                )

                # Option to regenerate PDF
                if st.button("Regenerate PDF"):
                    st.session_state.pdf_download_ready = False
                    st.rerun()
        else:
            st.warning(
                "PDF export is not available because ReportLab is not installed. "
                "To enable PDF exports, install ReportLab: pip install reportlab"
            )

        # Option to reset content
        if st.button("Reset Content"):
            st.session_state.system_response = ""
            st.session_state.pdf_download_ready = False
            st.session_state.pdf_data = None
            st.rerun()


# Main function
def main():
    """Main program execution"""
    st.set_page_config(
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Display appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        main_application()
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.pdf_download_ready = False
            st.session_state.pdf_data = None
            st.rerun()


# Run the application
if __name__ == "__main__":
    main()