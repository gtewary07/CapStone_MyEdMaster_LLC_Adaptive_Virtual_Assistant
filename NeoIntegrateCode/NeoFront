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

# Initialize FastAPI application
app = FastAPI()

# Load the tokenizer and the pre-trained classification model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)  # Load the tokenizer used for text-to-sequence conversion

model = load_model('question_classifier.keras')  # Load the pre-trained Keras model for question classification

# Load spaCy's pre-trained English language model for calculating semantic similarity
nlp = spacy.load("en_core_web_sm")

# Load the question-answer pairs from an Excel file
qa_pairs = pd.read_excel('qa_pairs.xlsx')  # This file should contain a 'Question' and 'Answer' column

# Define the structure of the incoming request using Pydantic
class QuestionInput(BaseModel):
    question: str  # The question that the user submits as input

# Define the structure of the response model using Pydantic
class PredictionOutput(BaseModel):
    prediction: list  # The model's output prediction vector
    system_response: str  # The system's response to the question
    similarity_score: float  # The similarity score between the question and the system's response

# Function to predict the category of a question using the trained classification model
def predict_question(question: str):
    sequences = tokenizer.texts_to_sequences([question])  # Convert the input question to a sequence of integers
    X_new = pad_sequences(sequences, maxlen=10)  # Pad the sequence to the expected input size for the model
    prediction = model.predict(X_new)[0]  # Get the model's prediction for the question
    return prediction.tolist()  # Return the prediction as a list for JSON serialization

# Function to retrieve an appropriate response based on exact matching from the QA pairs dataset
def get_system_response(question: str):
    # Search for an exact match for the question in the QA pairs dataset (case-insensitive)
    matching_row = qa_pairs[qa_pairs['Question'].str.lower() == question.lower()]
    if not matching_row.empty:  # If a match is found
        return matching_row.iloc[0]['Answer']  # Return the corresponding answer
    return "I'm sorry, I don't have a specific answer for that question."  # Default response if no match is found

# Function to calculate the semantic similarity between two pieces of text using spaCy
def calculate_similarity(text1: str, text2: str):
    doc1 = nlp(text1)  # Process the first text with spaCy NLP model
    doc2 = nlp(text2)  # Process the second text
    return doc1.similarity(doc2)  # Return the similarity score between the two processed texts

# Define a POST route to handle the question prediction request
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    question = input_data.question  # Extract the question from the input data
    prediction_vector = predict_question(question)  # Get the model's prediction for the question
    system_response = get_system_response(question)  # Get the system's response based on the question
    similarity_score = calculate_similarity(question, system_response)  # Calculate the similarity between the question and the system's response

    # Return a structured response containing the prediction vector, system response, and similarity score
    return {
        "prediction": prediction_vector,
        "system_response": system_response,
        "similarity_score": similarity_score
    }

# Run the FastAPI application using uvicorn when this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the app on all network interfaces at port 8000
