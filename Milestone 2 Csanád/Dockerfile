FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "distilbert_mrqa_preprocess.py"]