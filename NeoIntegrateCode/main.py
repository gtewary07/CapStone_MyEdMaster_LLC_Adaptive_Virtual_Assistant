import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import great_expectations as gx

app = FastAPI()

# Load the tokenizer and model
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('question_classifier.keras')
except FileNotFoundError:
    print("Tokenizer or model file not found. Make sure to train the model first.")
    tokenizer, model = None, None


class QuestionInput(BaseModel):
    question: str


class PredictionOutput(BaseModel):
    prediction: list


def weak_supervision_label(question):
    if not isinstance(question, str):
        return [0.33, 0.33, 0.33]

    question = question.lower()
    what_like = ["what", "which", "who", "where", "when"]
    how_like = ["how", "in what way", "by what means"]
    why_like = ["why", "for what reason", "how come"]

    label = np.zeros(3)
    weights = {"start": 1.5, "middle": 1.0, "end": 2.0}

    for i, category in enumerate([what_like, how_like, why_like]):
        for word in category:
            if word in question:
                if question.endswith(word):
                    label[i] += weights["end"]
                elif question.startswith(word):
                    label[i] += weights["start"]
                else:
                    label[i] += weights["middle"]

    if np.sum(label) > 0:
        label /= np.sum(label)
    else:
        label = np.array([0.33, 0.33, 0.33])

    return label.tolist()


def predict_question(question: str):
    sequences = tokenizer.texts_to_sequences([question])
    X_new = pad_sequences(sequences, maxlen=10)
    prediction = model.predict(X_new)[0]
    return prediction.tolist()


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    prediction_vector = predict_question(input_data.question)
    return {"prediction": prediction_vector}


@app.post("/weak_label", response_model=PredictionOutput)
async def weak_label(input_data: QuestionInput):
    label = weak_supervision_label(input_data.question)
    return {"prediction": label}


def process_data(data_path):
    # Implement your data processing logic here
    # This is a placeholder implementation
    data = pd.read_pickle(data_path)
    questions = data['question'].tolist()
    labels = data['label'].tolist()
    return questions, labels


def train_model(data_path, input_dim=1000, output_dim=64, input_length=10):
    questions, labels = process_data(data_path)

    tokenizer = Tokenizer(num_words=input_dim)
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    X = pad_sequences(sequences, maxlen=input_length)
    y = np.array(labels)

    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        LSTM(output_dim),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, verbose=2)

    # Save the model and tokenizer
    model.save('question_classifier.keras')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer


@app.post("/train")
async def train_model_endpoint(data_path: str):
    try:
        model, tokenizer = train_model(data_path)
        return {"message": "Model trained and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


def run_expectations(data_path):
    context = gx.get_context()
    expectation_suite_name = "question_classification_suite"

    try:
        suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)
    except:
        suite = context.add_expectation_suite(expectation_suite_name=expectation_suite_name)

    batch_request = gx.core.BatchRequest(
        datasource_name="my_datasource",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name=data_path,
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    validator.expect_column_values_to_not_be_null(column="question")
    validator.expect_column_values_to_be_of_type(column="question", type_="str")

    context.save_expectation_suite(suite)


@app.post("/validate_data")
async def validate_data(data_path: str):
    try:
        run_expectations(data_path)
        return {"message": "Data validation completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)