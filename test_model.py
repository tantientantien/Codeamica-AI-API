from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI(title="T5 Question Generation API")
model_path = "./t5_squad_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

class ContextInput(BaseModel):
    context: str

def generate_question_answer(context: str) -> str:
    input_text = "Generate a question and answer from the context: " + context
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)

    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/generate")
def generate(data: ContextInput):
    if not data.context:
        raise HTTPException(status_code=400, detail="Missing context")

    output_text = generate_question_answer(data.context)
    return {"generated_output": output_text}

@app.get("/")
def read_root():
    return {"message": "Welcome to the T5 Question Generation API"}