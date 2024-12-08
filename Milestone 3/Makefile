
# Check for an NVidia GPU
ifeq (, $(shell which nvidia-smi 2> /dev/null))
GPU=
else
GPU=--gpus all
endif
all: build

.PHONY: build
build:
	docker build -t distilbert-finetuning .

.PHONY: start
start:
	docker run ${GPU} --rm -it -p 8000:8000 -v "$${PWD}/training:/app/training" distilbert-finetuning:latest

.PHONY: train
train:
	docker run ${GPU} --rm -it -v "$${PWD}/training:/app/training" distilbert-finetuning:latest python training/training_script_2.py