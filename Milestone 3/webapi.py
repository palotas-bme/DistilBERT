#!/usr/bin/env python3

from pyexpat import model
import uvicorn
from questionanswerer import QuestionAnswerer

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str
    context: str | None = None

class Answer(BaseModel):
    context: str
    link: str
    answer: str

@app.post("/ask")
def answer(q: Question):
    model_answer = qa.answer_question(q.question, q.context)
    print(model_answer)
    print(len(model_answer))
    a = Answer(context=model_answer[0]["text"], link=model_answer[0]["link"], answer=model_answer[0]["answer"])
    return a


app.mount("/", StaticFiles(directory="ai-frontend/dist", html=True), name="frontend")

@app.get("/")
def read_root():
    return FileResponse('ai-frontend/dist/index.html')


qa = QuestionAnswerer()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
