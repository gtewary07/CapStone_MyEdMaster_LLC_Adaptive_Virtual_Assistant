import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


############################################
# Learning Report Functions
############################################

def generate_understanding_report(direct_answer_evaluations):
    """
    Generate a comprehensive understanding report based on user's answers to direct questions.
    """
    if not direct_answer_evaluations:
        return {
            'overall_score': 0,
            'overall_level': 0,
            'rationale_level': 0,
            'factual_level': 0,
            'procedural_level': 0,
            'strengths': [],
            'areas_for_improvement': [],
            'recommendations': []
        }

    # Calculate overall score and convert to level (1-5 scale)
    total_score = sum(eval_item.get('score', 0) for eval_item in direct_answer_evaluations)
    avg_score = total_score / len(direct_answer_evaluations)
    overall_level = min(5, max(1, round(avg_score / 20)))  # Convert 0-100 to 1-5

    # Group evaluations by question type and calculate average levels
    rationale_evals = [eval_item for eval_item in direct_answer_evaluations
                       if eval_item.get('question_type') == 'rationale']
    factual_evals = [eval_item for eval_item in direct_answer_evaluations
                     if eval_item.get('question_type') == 'factual']
    procedural_evals = [eval_item for eval_item in direct_answer_evaluations
                        if eval_item.get('question_type') == 'procedural']

    # Calculate average levels for each dimension
    rationale_level = calculate_dimension_level(rationale_evals)
    factual_level = calculate_dimension_level(factual_evals)
    procedural_level = calculate_dimension_level(procedural_evals)

    # Identify strengths and areas for improvement
    dimensions = [
        ('Rationale Understanding', rationale_level),
        ('Factual Knowledge', factual_level),
        ('Procedural Knowledge', procedural_level)
    ]

    strengths = [dim[0] for dim in dimensions if dim[1] >= 4]
    areas_for_improvement = [dim[0] for dim in dimensions if dim[1] <= 2]

    # Generate recommendations
    recommendations = generate_recommendations(
        rationale_level, factual_level, procedural_level, direct_answer_evaluations
    )

    return {
        'overall_score': avg_score,
        'overall_level': overall_level,
        'rationale_level': rationale_level,
        'factual_level': factual_level,
        'procedural_level': procedural_level,
        'strengths': strengths,
        'areas_for_improvement': areas_for_improvement,
        'recommendations': recommendations
    }


def calculate_dimension_level(evaluations):
    """Calculate the average level for a specific dimension."""
    if not evaluations:
        return 0

    # If we have understanding_level directly
    if all('understanding_level' in eval_item for eval_item in evaluations):
        total = sum(eval_item.get('understanding_level', 0) for eval_item in evaluations)
        return round(total / len(evaluations))

    # If we only have scores (0-100), convert to levels (1-5)
    total_score = sum(eval_item.get('score', 0) for eval_item in evaluations)
    avg_score = total_score / len(evaluations)
    return min(5, max(1, round(avg_score / 20)))  # Convert 0-100 to 1-5


def generate_recommendations(rationale_level, factual_level, procedural_level, evaluations):
    """Generate personalized recommendations based on performance."""
    recommendations = []

    # Add dimension-specific recommendations
    if rationale_level <= 3:
        recommendations.append("Focus on understanding the 'why' behind concepts - try to explain reasons and causes.")

    if factual_level <= 3:
        recommendations.append("Strengthen your factual knowledge by reviewing key definitions and concepts.")

    if procedural_level <= 3:
        recommendations.append("Practice applying concepts through step-by-step procedures and hands-on exercises.")

    # Add general recommendations
    if min(rationale_level, factual_level, procedural_level) <= 2:
        recommendations.append("Consider revisiting the core material before proceeding to more advanced topics.")

    # Add recommendations based on specific questions with low scores
    low_score_questions = [eval_item for eval_item in evaluations if eval_item.get('score', 0) < 60]
    if low_score_questions:
        recommendations.append(
            f"Review the topics related to: {', '.join([q['question'][:50] + '...' for q in low_score_questions[:2]])}")

    # Ensure we have at least one recommendation
    if not recommendations:
        if max(rationale_level, factual_level, procedural_level) >= 4:
            recommendations.append("You're doing well! Challenge yourself with more advanced material in this subject.")
        else:
            recommendations.append("Continue practicing across all dimensions for balanced improvement.")

    return recommendations


def create_understanding_radar_chart(report):
    """
    Create a radar chart visualizing understanding across different dimensions.
    Returns the chart as a base64 encoded image.
    """
    categories = ['Rationale', 'Factual', 'Procedural']
    values = [
        report.get('rationale_level', 0),
        report.get('factual_level', 0),
        report.get('procedural_level', 0)
    ]

    # Close the plot
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    values += values[:1]  # Close the loop

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Set the background color
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('#f0f2f6')

    # Draw one axis per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=14, fontweight='bold')

    # Draw ylabels (levels from 1 to 5)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], color="grey", size=12)
    plt.ylim(0, 5)

    # Plot data
    ax.plot(angles, values, linewidth=3, linestyle='solid', color='#1f77b4')

    # Fill area
    ax.fill(angles, values, '#1f77b4', alpha=0.25)

    # Add a title
    plt.title(f"Understanding Level Profile", size=18, color='black', y=1.1, fontweight='bold')

    # Add level descriptions
    level_descriptions = {
        1: "Minimal",
        2: "Basic",
        3: "Moderate",
        4: "Good",
        5: "Excellent"
    }

    # Add explanation text
    fig.text(0.5, 0.01,
             "This chart shows your understanding level in three dimensions.\nHigher values (closer to 5) indicate stronger understanding.",
             ha='center', fontsize=12)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    plt.close(fig)

    return img_str


def create_progress_chart(scores_history, metric='level'):
    """
    Create a line chart showing progress over time.
    Returns the chart as a base64 encoded image.
    """
    if not scores_history:
        # Return empty chart if no history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No history data available", ha='center', va='center', fontsize=14)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    # Create a DataFrame from the history
    df = pd.DataFrame(scores_history)

    # Ensure we have a timestamp column
    if 'timestamp' not in df.columns:
        # Create dummy timestamps if none exist
        now = datetime.now()
        df['timestamp'] = [now - timedelta(days=i) for i in range(len(df) - 1, -1, -1)]

    # Convert timestamps to datetime if they're strings
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Choose the metric to plot
    if metric == 'level' and 'level' in df.columns:
        y_values = df['level']
        title = "Understanding Level Progress"
        ylabel = "Level (1-5)"
        ylim = (0, 5.5)
    else:
        y_values = df['score']
        title = "Score Progress"
        ylabel = "Score (0-100)"
        ylim = (0, 105)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(df['timestamp'], y_values, marker='o', linestyle='-', linewidth=2, markersize=8)

    # Set the title and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(ylim)

    # Format the grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Format the date on the x-axis
    fig.autofmt_xdate()

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    plt.close(fig)

    return img_str


def display_learning_report(direct_answer_evaluations, scores_history):
    """
    Display a comprehensive learning report in the Streamlit app.
    """
    # Generate the understanding report
    report = generate_understanding_report(direct_answer_evaluations)

    # Create the radar chart
    radar_chart_img = create_understanding_radar_chart(report)

    # Create the progress chart
    progress_chart_img = create_progress_chart(scores_history, metric='level')

    # Display the report
    st.header("📊 Learning Progress Report")

    # Display overall score and level
    st.subheader("Overall Understanding")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Score", f"{report['overall_score']:.1f}/100")
    with col2:
        level_descriptions = {
            0: "Not Assessed",
            1: "Minimal",
            2: "Basic",
            3: "Moderate",
            4: "Good",
            5: "Excellent"
        }
        st.metric("Understanding Level", f"{report['overall_level']}/5 - {level_descriptions[report['overall_level']]}")

    # Display the radar chart
    st.subheader("Understanding Profile")
    st.image(f"data:image/png;base64,{radar_chart_img}", use_column_width=True)

    # Display dimension scores
    st.subheader("Dimension Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rationale Understanding", f"{report['rationale_level']}/5")
        st.caption("How well you understand 'why' things work")
    with col2:
        st.metric("Factual Knowledge", f"{report['factual_level']}/5")
        st.caption("Your grasp of core facts and concepts")
    with col3:
        st.metric("Procedural Knowledge", f"{report['procedural_level']}/5")
        st.caption("Your ability to apply or implement")

    # Display strengths and areas for improvement
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 Strengths")
        if report['strengths']:
            for strength in report['strengths']:
                st.markdown(f"- {strength}")
        else:
            st.markdown("Keep practicing to develop strengths.")

    with col2:
        st.subheader("🔍 Areas for Improvement")
        if report['areas_for_improvement']:
            for area in report['areas_for_improvement']:
                st.markdown(f"- {area}")
        else:
            st.markdown("You're doing well across all areas!")

    # Display personalized recommendations
    st.subheader("📝 Personalized Recommendations")
    for i, recommendation in enumerate(report['recommendations']):
        st.markdown(f"{i + 1}. {recommendation}")

    # Display progress over time
    if scores_history:
        st.subheader("📈 Progress Over Time")
        st.image(f"data:image/png;base64,{progress_chart_img}", use_column_width=True)

    # Add a download button for the report
    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=f"Placeholder for PDF report. In a real implementation, this would be a generated PDF.",
        file_name=f"learning_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )

    return report