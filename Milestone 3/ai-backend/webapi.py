#!/usr/bin/env python3

import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Question(BaseModel):
    question: str
    context: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI example!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
