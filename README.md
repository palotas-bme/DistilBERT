# Deep Learning assignment - Question answering with DistilBERT

Team name: **Tensor Titans**

## Team members

- Péter M. Palotás - K8GD4Q
- Csanád Mihály Skribanek - G0OTOY

## Project description

Read the paper of the DistilBERT Model. This is a distilled version of BERT that is smaller, quicker, cheaper, and lighter than the original BERT. (Compared to bert-base-uncased, it runs 60% faster and uses 40% less parameters while maintaining over 95% of BERT's performance on the GLUE language understanding benchmark.)

You can find the source of the model here: <https://huggingface.co/distilbert-base-uncased-distilled-squad>
This model is a DistilBERT-base-uncased fine-tune checkpoint that was refined using knowledge distillation on SQuAD v1.1. Ideally, this should run in Colab.
You can also use parameter-efficient methods (low-rank adaptation, quantization, etc.)

Finetune the model for question answering task. Build the inference pipeline, report results for the pretrained model and for the fine-tuned version, as well.
If you're up for a challenge, you can find other databases, or translate them to Hungarian and see how your model performs on Hungarian questions.

We use the MRQA dataset to finetune our model which is a dataset for training LLMs for extractive question answering tasks. It includes multiple datasets (for example: SQUAD, TriviaQA, NaturalQA) in a uniform format. 

Related GitHub repository:
<https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation>

Related paper:
<https://arxiv.org/abs/1910.01108>

## Build

```sh
make
```

## Run

```sh
docker run distilbert-finetuning:latest
```
