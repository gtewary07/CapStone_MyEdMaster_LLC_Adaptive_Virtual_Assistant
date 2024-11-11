import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Load the CSV file
qa_pairs = pd.read_csv('qa_pairs.csv')

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('question_classifier.keras')


def get_most_similar_question(user_question, questions):
    user_vec = nlp(user_question).vector
    question_vecs = np.array([nlp(str(q)).vector for q in questions])
    similarities = cosine_similarity([user_vec], question_vecs)[0]
    max_similarity = np.max(similarities)
    most_similar_index = np.argmax(similarities)
    return max_similarity, most_similar_index


class QuestionInput(BaseModel):
    question: str


class PredictionOutput(BaseModel):
    prediction: list
    system_response: str
    similarity_score: float


class UnderstandingInput(BaseModel):
    original_question: str
    system_response: str
    user_answer: str


class UnderstandingOutput(BaseModel):
    understanding_score: float


def get_system_response(question: str) -> tuple:
    max_similarity, most_similar_index = get_most_similar_question(question, qa_pairs['Question'])
    if max_similarity > 0.7:
        return qa_pairs.iloc[most_similar_index]['System_Response'], max_similarity
    else:
        return "I'm sorry, I don't have a specific answer for that question.", max_similarity


def predict_question(question: str):
    sequences = tokenizer.texts_to_sequences([question])
    X_new = pad_sequences(sequences, maxlen=10)
    prediction = model.predict(X_new)[0]
    return prediction.tolist()


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    try:
        question = input_data.question
        print(f"Received question: {question}")

        prediction_vector = predict_question(question)
        print(f"Prediction vector: {prediction_vector}")

        system_response, similarity_score = get_system_response(question)
        print(f"System response: {system_response}")
        print(f"Similarity score: {similarity_score}")

        return {
            "prediction": prediction_vector,
            "system_response": system_response,
            "similarity_score": similarity_score
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/assess_understanding", response_model=UnderstandingOutput)
async def assess_understanding(input_data: UnderstandingInput):
    try:
        original_question = input_data.original_question
        system_response = input_data.system_response
        user_answer = input_data.user_answer

        # Combine original question and system response
        reference_text = f"{original_question} {system_response}"

        # Calculate similarity between reference text and user answer
        reference_vec = nlp(reference_text).vector
        user_vec = nlp(user_answer).vector

        similarity_score = cosine_similarity([reference_vec], [user_vec])[0][0]

        # Normalize the score to a 0-100 scale
        understanding_score = similarity_score * 100

        return {"understanding_score": round(understanding_score, 2)}

    except Exception as e:
        print(f"Error in assessing understanding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)