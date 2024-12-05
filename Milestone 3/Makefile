all: build

.PHONY: build
build:
	docker build -t distilbert-finetuning .

.PHONY: start
start:
	docker run --rm -it -p 8000:8000 distilbert-finetuning