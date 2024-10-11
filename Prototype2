import spacy
from spacy.matcher import PhraseMatcher
import random

nlp = spacy.load("en_core_web_sm")

questions = [
    "What are the primary hallmarks of aging?",
    "How do antagonistic hallmarks of aging differ from primary hallmarks?",
    "What role does cellular senescence play in aging?",
    "How does stem cell exhaustion contribute to aging?",
    "What is a promising approach to potentially reverse aspects of aging?"
]

answers = [
    "The primary hallmarks of aging are genomic instability, telomere attrition, epigenetic alterations, and loss of proteostasis. These are considered unequivocally negative and accumulate with time.",
    "Antagonistic hallmarks like cellular senescence and deregulated nutrient sensing have opposite effects depending on their intensity - at low levels they can be beneficial, but at high levels they become deleterious.",
    "Cellular senescence can be both beneficial and detrimental - it helps suppress tumors, but excessive accumulation of senescent cells can promote aging and age-related diseases.",
    "Stem cell exhaustion leads to a decline in the regenerative potential of tissues, which is one of the most obvious characteristics of aging. It results from multiple types of aging-associated cellular damage.",
    "Rejuvenation of stem cells and restoration of the stem cell niche show promise for reversing certain aging phenotypes at the organismal level."
]


def display_qa(index):
    print(f"Question: {questions[index]}")
    print(f"Answer: {answers[index]}")


def get_feedback():
    understanding = input("How well did you understand the topic? (1-10): ")
    try:
        understanding = int(understanding)
        if understanding < 1 or understanding > 10:
            raise ValueError
    except ValueError:
        print("Please enter a number between 1 and 10.")
        return get_feedback()

    if understanding <= 3:
        return "Low"
    elif understanding <= 7:
        return "Medium"
    else:
        return "High"


def rate_similarity(user_answer, correct_answer):
    user_doc = nlp(user_answer.lower())
    correct_doc = nlp(correct_answer.lower())
    similarity = user_doc.similarity(correct_doc)
    return similarity


quiz_questions = [
    {
        "question": "Which of the following is NOT a primary hallmark of aging?",
        "options": ["Genomic instability", "Telomere attrition", "Cellular senescence", "Loss of proteostasis"],
        "correct": "Cellular senescence"
    },
    {
        "question": "What characterizes antagonistic hallmarks of aging?",
        "options": ["Always beneficial", "Always harmful", "Beneficial at low levels, harmful at high levels",
                    "No effect on aging"],
        "correct": "Beneficial at low levels, harmful at high levels"
    },
    {
        "question": "What is a potential benefit of cellular senescence?",
        "options": ["Promotes aging", "Suppresses tumors", "Increases tissue regeneration",
                    "Enhances nutrient sensing"],
        "correct": "Suppresses tumors"
    },
    {
        "question": "What is a consequence of stem cell exhaustion?",
        "options": ["Increased tissue regeneration", "Decreased tissue regeneration", "Enhanced proteostasis",
                    "Reduced genomic instability"],
        "correct": "Decreased tissue regeneration"
    },
    {
        "question": "What approach shows promise for reversing aging phenotypes?",
        "options": ["Increasing cellular senescence", "Promoting telomere attrition", "Rejuvenation of stem cells",
                    "Enhancing genomic instability"],
        "correct": "Rejuvenation of stem cells"
    }
]


def check_different_answer(user_answer, correct_answer):
    return user_answer.lower() != correct_answer.lower()


def get_user_explanation(correct_answer):
    while True:
        user_explanation = input("Please explain the answer in your own words: ")
        if not check_different_answer(user_explanation, correct_answer):
            print("Restate the answer in your own words.")
        else:
            return user_explanation


def display_quiz(index):
    quiz = quiz_questions[index]
    print(f"\nQuiz Question: {quiz['question']}")
    for i, option in enumerate(quiz['options'], 1):
        print(f"{i}. {option}")

    user_answer = input("Enter the number of your answer: ")
    try:
        user_answer = int(user_answer)
        if user_answer < 1 or user_answer > len(quiz['options']):
            raise ValueError
    except ValueError:
        print("Please enter a valid option number.")
        return display_quiz(index)

    if quiz['options'][user_answer - 1] == quiz['correct']:
        print("Correct!")
    else:
        print(f"Incorrect. The correct answer is: {quiz['correct']}")


def main():
    for i in range(len(questions)):
        display_qa(i)
        feedback = get_feedback()
        print(f"Understanding level: {feedback}")

        user_explanation = get_user_explanation(answers[i])
        similarity = rate_similarity(user_explanation, answers[i])
        print(f"Similarity to correct answer: {similarity:.2f}")

        display_quiz(i)

        print("\n---\n")


if __name__ == "__main__":
    main()
