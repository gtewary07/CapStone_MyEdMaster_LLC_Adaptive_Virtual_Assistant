import numpy as np
import spacy
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from enum import Enum


class QuestionType(Enum):
    FACTUAL = 0
    PROCEDURAL = 1
    CAUSAL = 2


class ResponseType(Enum):
    INFORMATIVE = 0
    REAL_WORLD_EXAMPLE = 1
    GOAL_OR_OUTCOME = 2
    CAUSAL_PRINCIPLE = 3


class User():
    '''
    Types of Questions (Rows):
    0 - Factual 
    1 - Precedure
    2 - Causal

    Types of Responses (Columns):
    0 - Informative
    1 - Real World Example
    2 - Goal or Outcome
    3 - Causal Principle 
    '''

    def __init__(self):
        # Necessary attributes
        self.nlp = spacy.load("en_core_web_sm")
        self.weights = np.array([
            [0.85, 0.15, 0, 0],
            [0.53, 0.33, 0.12, 0.02],
            [0.69, 0.103, 0.103, 0.103]])

        # Initialize ML components
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.question_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.response_selector = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

        self._learning_rate = 0.1  # In future, can make adaptive

        # Extra data
        self._num_questions = 0

    def classify_question(self, question):
        doc = self.nlp(question)

        features = [
            len(doc),
            len([token for token in doc if token.pos_ == "VERB"]),
            len([token for token in doc if token.dep_ == "ROOT"]),
            len([ent for ent in doc.ents])
        ]

        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        question_type = self.question_classifier.predict(features)[0]

        question_array = np.zeros(len(QuestionType))
        question_array[question_type] = 1

        return question_array, QuestionType(question_type)

    def incoming_question(self, question_array,question):
        # question_array is 1x3 weights for Fact, Prec, Caus
        # 1x4 weights for Info, RW, Goal, Caus
        # Matrix Multiplication weighs response matrix weights by appropriate question type
        self.num_questions += 1
        self._question_array = self.classify_question(question)
        self._answer_array = self._question_array @ self.weights
        return self._answer_array

        self._answer_array = question_array @ self.weights
        return self._answer_array

    def select_response_type(self, question, answer_array):
        # Combine question text and answer_array as features
        doc = self.nlp(question)
        question_vector = self.vectorizer.transform([question]).toarray()
        combined_features = np.hstack((question_vector, answer_array.reshape(1, -1)))

        # Use the response selector to choose response type
        response_type = self.response_selector.predict(combined_features)[0]
        return response_type

    def update_weights(self, feedback: int):  # Feedback is 1 or 0
        # For future calculations
        if feedback == 0:
            feedback = -1

        # Changes dimensionality of original arrays for multiplication
        # (4 -> 4x1)
        response_transformed = self._answer_array[:, np.newaxis].T
        # (3 -> 1x3)
        question_transformed = self._question_array[:, np.newaxis]

        # Create a matrix weighted by question type and response type
        self._response_matrix = question_transformed @ response_transformed  # 3x4

        # Changes direct based on user feedback
        self._response_matrix_transformed = self._response_matrix * self._user_response

        # Fraction added to original weights
        self.weights = self.weights + self._response_matrix_transformed * self.learning_rate

        # Makes rows sum to 1
        self._clean_weights()

    def _clean_weights(self):
        # 1x3 array of row minimums, 0 if minimum is positive
        row_minimums = np.min(self.weights, axis=1).clip(max=0)[:, np.newaxis]

        # Makes all weights positive, adds buffer so no weights are zero
        self.weights -= row_minimums - 0.001

        # 1x3 array of row sums
        weights_row_sums = np.sum(self.weights, axis=1)[:, np.newaxis]

        # Divides each row by it's respective sum
        proportioned_weights = self.weights / weights_row_sums

        # Rounds weights to 3 decimal points
        self.weights = np.round(proportioned_weights, 3)
        
