# NeoMain.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import spacy
from groq import Groq
# At the top of NeoMain.py
import os
os.environ["GROQ_API_KEY"] = ""

app = FastAPI()

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize Groq client with API key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)



class QuestionInput(BaseModel):
    question: str
    age: Optional[int] = None
    knowledge_level: Optional[str] = None


class PredictionOutput(BaseModel):
    system_response: str


class UnderstandingInput(BaseModel):
    original_question: str
    system_response: str
    user_answer: str


class UnderstandingOutput(BaseModel):
    understanding_score: float


def get_personalized_prompt(question: str, age: Optional[int], knowledge_level: Optional[str]) -> str:
    base_prompt = f"Question: {question}\n"
    if age:
        base_prompt += f"Please provide an answer suitable for someone who is {age} years old. "
    if knowledge_level:
        base_prompt += f"The person has a {knowledge_level} level of knowledge in this subject. "
    base_prompt += "Please explain in a clear and engaging way."
    return base_prompt


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: QuestionInput):
    try:
        prompt = get_personalized_prompt(
            input_data.question,
            input_data.age,
            input_data.knowledge_level
        )

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides personalized responses based on the user's age and knowledge level."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        system_response = completion.choices[0].message.content

        return {
            "system_response": system_response
        }

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/assess_understanding", response_model=UnderstandingOutput)
async def assess_understanding(input_data: UnderstandingInput):
    try:
        reference_text = f"{input_data.original_question} {input_data.system_response}"
        reference_vec = nlp(reference_text).vector
        user_vec = nlp(input_data.user_answer).vector

        from sklearn.metrics.pairwise import cosine_similarity
        similarity_score = cosine_similarity([reference_vec], [user_vec])[0][0]
        understanding_score = similarity_score * 100

        return {"understanding_score": round(understanding_score, 2)}

    except Exception as e:
        print(f"Error in assessing understanding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
