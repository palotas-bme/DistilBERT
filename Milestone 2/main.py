#!/usr/bin/env python3

from datasets import load_dataset
from transformers import AutoTokenizer
import json
from transformers import pipeline
import torch
from transformers import DistilBertForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
import os
import wandb
from trl import SFTTrainer
import evaluate
import numpy as np
from tqdm.auto import tqdm
import collections
from torch.utils.data import DataLoader

# Loading the dataset
# The MRQA dataset is included in huggingface's datasets library, so we just have to load it
# Loading dataset (smaller fraction than in the final becasue had to train on local GPU)
mrqa = load_dataset("mrqa", split="train[:5%]")
# Creating the train-test-validation split
mrqa = mrqa.train_test_split(test_size=0.2)
mrqa["train"] = mrqa["train"].train_test_split(test_size=0.2)
mrqa["val"] = mrqa["train"]["test"]
mrqa["train"] = mrqa["train"]["train"]

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad")

max_length = 384
stride = 128

# Preprocessing function for training
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["detected_answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["char_spans"][0]["start"][0]
        end_char = answer["char_spans"][0]["end"][0]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Preprocessing function for the validation
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["qid"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


# Defining data collator
data_collator = DefaultDataCollator()

# Configuring parameters for the quantation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

# Configuring parameters of the low-rank adaptation
peft_config = LoraConfig(
    lora_alpha=6,
    lora_dropout=0.15,
    r=2,
    bias="none",
    task_type="QUESTION_ANS",
    target_modules=["q_lin", "k_lin", "v_lin", "ffn.lin1", "ffn.lin2", "attention.out_proj"])

# Loading baseline model: DistilBert finetuned on Squadn dataset
model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad",
                                                       quantization_config=bnb_config,
                                                       device_map={"": 0})

# Defining some parameters for computing the evaluation metrics
metric = evaluate.load("squad")
n_best = 20
max_answer_length = 30

# Evaluation funnction with default squad metrics (exact match and f1 score)
def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["qid"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
    {"id": ex["qid"], "answers": [{"text": ans, "answer_start": 0} for ans in ex["answers"]]}
    for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


# Function for doing the complete evaluation on the preprocessed evaluation set
def eval_function(tokenized_eval, batch_size=64):
    # Remove unneccesary columns from evaluation set and convert it to torch tensors
    eval_set_for_model = tokenized_eval.remove_columns(["example_id", "offset_mapping"])
    eval_set_for_model.set_format("torch")

    # Evaluate on GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create a DataLoader for the evaluation set
    eval_loader = DataLoader(eval_set_for_model, batch_size=batch_size)

    # Log start and end logits
    all_start_logits = []
    all_end_logits = []

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    # Concatennate start and end logit labels
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    return compute_metrics(start_logits, end_logits, tokenized_eval, mrqa["test"])

# Calculating evaluation metrics before further finetuning
pre_training_metrics = eval_function(tokenized_eval)
print(f"Exact match before finetuning: {pre_training_metrics['exact_match']}\nF1 score before finetuning: {pre_training_metrics['f1']}")

# Defining training parameters
output_dir_name = "trial_run_local"

# Configuring parameters of the low-rank adaptation
peft_config = LoraConfig(
    lora_alpha=6,
    lora_dropout=0.15,
    r=2,
    bias="none",
    task_type="QUESTION_ANS",
    target_modules=["q_lin", "k_lin", "v_lin", "ffn.lin1", "ffn.lin2", "attention.out_proj"])

# Parameters will be later adjusted, this is only to ensure that training pipeline works as intended
training_args = TrainingArguments(
    output_dir=output_dir_name,
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='new_dir',
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mrqa["train"],
    eval_dataset=tokenized_mrqa["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    peft_config=peft_config
)

# Training the model
wandb_project = input("Wandb project: ")
wandb_entity = input("Wandb entity: ")
wandb.init(project=wandb_project, entity=wandb_entity)
trainer.train()
wandb.finish()



post_training_metrics = eval_function(tokenized_eval)
print(f"Exact match after finetuning: {post_training_metrics['exact_match']}\nF1 score after finetuning: {post_training_metrics['f1']}")
