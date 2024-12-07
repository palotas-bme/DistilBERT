#!/usr/bin/env python3

from ast import mod
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
    # assert len(model_answer) > 0
    if len(model_answer) == 0:
        # https://users.ece.cmu.edu/~gamvrosi/thelastq.html
        return [Answer(context="", link="", answer="INSUFFICIENT DATA FOR MEANINGFUL ANSWER")]

    a = []
    for ans in model_answer:
        a.append(Answer(context=ans["text"], link=ans["link"], answer=ans["answer"]))
    return a


app.mount("/", StaticFiles(directory="ai-frontend/dist", html=True), name="frontend")

@app.get("/")
def read_root():
    return FileResponse('ai-frontend/dist/index.html')


qa = QuestionAnswerer()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
