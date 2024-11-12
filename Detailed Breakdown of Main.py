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

# Initialize FastAPI application
app = FastAPI()

# Load the tokenizer and model if they exist
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)  # Load pre-trained tokenizer
    model = load_model('question_classifier.keras')  # Load pre-trained model
except FileNotFoundError:
    # Handle missing files error if tokenizer or model is not found
    print("Tokenizer or model file not found. Make sure to train the model first.")
    tokenizer, model = None, None

# Pydantic models for input and output validation
class QuestionInput(BaseModel):
  # Accepts a 'question' of type string as input
    question: str  

class PredictionOutput(BaseModel):
  # Output will be a list of predicted values
    prediction: list  

# Function to generate weak supervision labels based on keywords
def weak_supervision_label(question):
    if not isinstance(question, str):
        # Return default label if question is not a string
        return [0.33, 0.33, 0.33]  

    question = question.lower()  # Convert question to lowercase
    # Define keyword categories for different question types
    what_like = ["what", "which", "who", "where", "when"]
    how_like = ["how", "in what way", "by what means"]
    why_like = ["why", "for what reason", "how come"]

    # Initialize the label vector for 3 categories
    label = np.zeros(3)  
    # Define keyword position weights
    weights = {"start": 1.5, "middle": 1.0, "end": 2.0}  

    # Check each category for matching keywords in the question
    for i, category in enumerate([what_like, how_like, why_like]):
        for word in category:
            if word in question:
                # Assign weights based on keyword position (start, middle, end)
                if question.endswith(word):
                    label[i] += weights["end"]
                elif question.startswith(word):
                    label[i] += weights["start"]
                else:
                    label[i] += weights["middle"]

    # Normalize the label to ensure it sums to 1
    if np.sum(label) > 0:
        label /= np.sum(label)
    else:
        # Default if no match is found
        label = np.array([0.33, 0.33, 0.33])  

    # Return the label as a list
    return label.tolist()  

# Function to make a prediction based on the model
def predict_question(question: str):
    # Convert question to sequence of integers
    sequences = tokenizer.texts_to_sequences([question])  
    # Pad sequence to a fixed length
    X_new = pad_sequences(sequences, maxlen=10)  
    # Make the prediction
    prediction = model.predict(X_new)[0]  
    # Return the prediction as a list
    return prediction.tolist()  

# FastAPI endpoint to predict the class of a question
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    # Ensure model and tokenizer are loaded before predicting
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    # Get prediction
    prediction_vector = predict_question(input_data.question)  
    # Return prediction as JSON response
    return {"prediction": prediction_vector}  

# FastAPI endpoint for weak supervision labeling of a question
@app.post("/weak_label", response_model=PredictionOutput)
async def weak_label(input_data: QuestionInput):
    # Get weak supervision label
    label = weak_supervision_label(input_data.question)  
    # Return label as JSON response
    return {"prediction": label} 

# Function to process the dataset for training
def process_data(data_path):
    # Placeholder data processing logic (e.g., reading from a pickle file)
    data = pd.read_pickle(data_path)
    # Extract questions
    questions = data['question'].tolist()
    # Extract labels
    labels = data['label'].tolist() 
    # Return questions and labels
    return questions, labels  
  
# Function to train the model
def train_model(data_path, input_dim=1000, output_dim=64, input_length=10):
   # Get data for training
    questions, labels = process_data(data_path) 

    # Tokenize the questions
    tokenizer = Tokenizer(num_words=input_dim)
    # Fit tokenizer on training data
    tokenizer.fit_on_texts(questions) 
    # Convert questions to sequences
    sequences = tokenizer.texts_to_sequences(questions)  
    # Pad sequences
    X = pad_sequences(sequences, maxlen=input_length) 
    # Convert labels to a numpy array
    y = np.array(labels)  

    # Define the model architecture (LSTM for text classification)
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        # LSTM layer for sequence processing
        LSTM(output_dim), 
        # Dropout layer for regularization
        Dropout(0.5),  
        # Fully connected layer
        Dense(32, activation='relu'),  
        # Output layer for 3 classes
        Dense(3, activation='softmax')  
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    # Train the model
    model.fit(X, y, epochs=10, verbose=2)  

    # Save the trained model and tokenizer for future use
    # Save the model
    model.save('question_classifier.keras')  
    with open('tokenizer.pkl', 'wb') as f:
        # Save the tokenizer
        pickle.dump(tokenizer, f)  
    # Return the trained model and tokenizer
    return model, tokenizer  

# FastAPI endpoint to trigger model training
@app.post("/train")
async def train_model_endpoint(data_path: str):
    try:
        # Train the model
        model, tokenizer = train_model(data_path)  
        # Return success message
        return {"message": "Model trained and saved successfully"}  
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")  
      
# Function to run data validation using Great Expectations
def run_expectations(data_path):
    # Get Great Expectations context
    context = gx.get_context()  
    # Define expectation suite name
    expectation_suite_name = "question_classification_suite"  

    try:
        # Load the suite
        suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)  
    except:
        # Create suite if not found
        suite = context.add_expectation_suite(expectation_suite_name=expectation_suite_name)  

    # Define a batch request for data validation
    batch_request = gx.core.BatchRequest(
        datasource_name="my_datasource",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name=data_path,
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    # Define expectations for the 'question' column
    validator.expect_column_values_to_not_be_null(column="question")
    validator.expect_column_values_to_be_of_type(column="question", type_="str")

    # Save the validation suite
    context.save_expectation_suite(suite)  

# FastAPI endpoint to trigger data validation
@app.post("/validate_data")
async def validate_data(data_path: str):
    try:
        # Run the data validation function
        run_expectations(data_path)  
        # Return success message
        return {"message": "Data validation completed"}  
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")  

# Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)  
