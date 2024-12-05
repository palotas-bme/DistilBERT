FROM node:23-alpine AS builder
LABEL stage=builder
WORKDIR /ai-frontend
COPY ai-frontend /ai-frontend
RUN rm -rf /ai-frontend/dist
RUN npm install -g pnpm
RUN pnpm install
RUN pnpm build


FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY --from=builder /ai-frontend/dist /app/ai-frontend/dist
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk;nltk.download('punkt_tab');nltk.download('averaged_perceptron_tagger_eng')"

CMD ["python", "webapi.py"]