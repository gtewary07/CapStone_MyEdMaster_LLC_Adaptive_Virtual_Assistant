import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def introduction():
    print("Welcome to the Algebra 1 Assessment System!")
    print("This assessment will cover three main areas:")
    print("1. Fact-based questions about algebraic concepts")
    print("2. Strategy questions on solving algebraic problems")
    print("3. Problem-solving questions to apply your knowledge")
    print("Let's begin!\n")
    print("In algebra, we often encounter expressions and equations that involve unknown quantities. These unknowns are represented by symbols, typically letters, which we call variables. Understanding how to work with variables is fundamental to solving algebraic problems. ")
    print("When faced with an equation, our goal is usually to find the value of the variable that makes the equation true. This process often involves isolating the variable on one side of the equation by performing inverse operations on both sides. ")
    print("For example, we might need to subtract a constant term or divide by a coefficient to get the variable by itself. Once isolated, we can determine the variable's value. This technique is particularly useful when dealing with linear equations.")
    print(" By systematically applying these principles, we can solve a wide range of algebraic problems, from simple equations to more complex real-world applications.")


# Define the questions and model answers
questions_df = pd.DataFrame({
    'question': [
        "What is a variable?",
        "How do you isolate a variable in a linear equation?",
        "Solve for x: 2x + 5 = 13",
        "What is a square root?",
        "How do you combine like terms?"
    ],
    'answer': [
        "A variable is a symbol (usually a letter) that represents an unknown value.",
        "To isolate a variable, perform inverse operations on both sides of the equation to move all terms with the variable to one side and all other terms to the other side.",
        "x = 4",
        "A square root of a number is a value that, when multiplied by itself, gives the number.",
        "Combine like terms by adding or subtracting the coefficients of terms with the same variables and exponents."
    ],
    'type': ['factual', 'procedural', 'problem-solving', 'factual', 'procedural']
})

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
answer_vectors = vectorizer.fit_transform(questions_df['answer'])


def extract_numbers(text):
    return [float(num) for num in re.findall(r'\d+(?:\.\d+)?', text)]


def calculate_proximity(correct, student):
    if correct == 0:  # Avoid division by zero
        return 100 if student == 0 else 0

    percentage_diff = abs(student - correct) / correct * 100
    return max(0, 100 - percentage_diff)  # Cap at 0% minimum


def evaluate_answer(student_answer, model_answer, question_type):
    if question_type == 'problem-solving':
        correct_numbers = extract_numbers(model_answer)
        student_numbers = extract_numbers(student_answer)

        if len(correct_numbers) == len(student_numbers):
            proximities = [calculate_proximity(c, s) for c, s in zip(correct_numbers, student_numbers)]
            return np.mean(proximities) / 100  # Convert to 0-1 scale
        else:
            return 0  # If the number of extracted numbers doesn't match, consider it incorrect
    else:
        student_vector = vectorizer.transform([student_answer])
        model_vector = vectorizer.transform([model_answer])
        similarity = cosine_similarity(student_vector, model_vector)[0][0]
        return similarity


def run_assessment():
    scores = []
    student_answers = []

    print("Welcome to the Algebra 1 Assessment System!")
    print("Please answer the following questions:\n")

    for _, row in questions_df.iterrows():
        print(f"Question ({row['type']}): {row['question']}")
        student_answer = input("Your answer: ")
        student_answers.append(student_answer)

        similarity = evaluate_answer(student_answer, row['answer'], row['type'])
        scores.append(similarity)

        print(f"Answer similarity: {similarity:.2f}\n")

    return scores, student_answers


def analyze_performance(scores):
    knowledge_types = questions_df['type'].unique()

    print("\nPerformance Analysis:")
    for k_type in knowledge_types:
        type_scores = [s for s, t in zip(scores, questions_df['type']) if t == k_type]
        print(f"{k_type.capitalize()} questions: {np.mean(type_scores):.2f}")

    overall_score = np.mean(scores)
    print(f"\nOverall performance: {overall_score:.2f}")

    # Calculate correlation between knowledge components and problem-solving
    knowledge_scores = [s for s, t in zip(scores, questions_df['type']) if t != 'problem-solving']
    problem_solving_scores = [s for s, t in zip(scores, questions_df['type']) if t == 'problem-solving']

    if problem_solving_scores:
        knowledge_score = np.mean(knowledge_scores)
        problem_solving_score = np.mean(problem_solving_scores)


def main():
    introduction()
    scores, student_answers = run_assessment()
    analyze_performance(scores)


if __name__ == "__main__":
    main()
