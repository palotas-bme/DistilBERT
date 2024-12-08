#!/usr/bin/env python3
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DistilBERT question answerer", description="""
DistilBERT question answerer Deeplearning assignment
                                     """)
    parser.add_argument("-m", "--model_path", default="training/best models/2024-12-08_22-18-27_best_model")
    args = parser.parse_args()

    # We import everything after parsing the arguments so when the help is called, we don't have to wait for the imports. They can take some time.
    import uvicorn
    import datetime

    from questionanswerer import QuestionAnswerer

    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    app = FastAPI()

    class Question(BaseModel):
        """Define the schema for a question input using Pydantic"""
        question: str
        context: str | None = None  # pylint: disable=E1131

    class Answer(BaseModel):
        """Define the schema for an answer using Pydantic"""
        text: str | None = None  # pylint: disable=E1131
        link: str | None = None  # pylint: disable=E1131
        answer: str
        score: float | None = None  # pylint: disable=E1131

    class Rating(BaseModel):
        """Define the schema for rating using Pydantic"""
        answer: Answer
        rating: int

    @app.post("/ask")
    def answer(q: Question):
        model_answer = qa.answer_question(q.question, q.context)
        if len(model_answer) == 0:
            # https://users.ece.cmu.edu/~gamvrosi/thelastq.html
            return [Answer(text="", link="", answer="INSUFFICIENT DATA FOR MEANINGFUL ANSWER")]

        a = []
        for ans in model_answer:
            a.append(Answer.model_validate(ans))

        return a

    @app.post("/rate")
    def rate(r: Rating):
        # Save ratings in separate JSON files.
        # In a real environment the ratings needs to be stored and processed some other way.
        with open(f"./question-ratings/question-{datetime.datetime.now().isoformat()}.json", encoding="utf-8", mode="w") as f:
            f.write(r.model_dump_json())

    app.mount("/", StaticFiles(directory="ai-frontend/dist", html=True), name="frontend")

    @app.get("/")
    def read_root():
        return FileResponse('ai-frontend/dist/index.html')

    qa = QuestionAnswerer(model_path=args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)
