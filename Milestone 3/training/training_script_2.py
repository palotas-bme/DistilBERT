from training_utils import *
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
mrqa = load_dataset("mrqa", split="train[:40%]")
# Creating the train-test-validation split
mrqa = mrqa.train_test_split(test_size=0.2)
mrqa["train"] = mrqa["train"].train_test_split(test_size=0.2)
mrqa["val"] = mrqa["train"]["test"]
mrqa["train"] = mrqa["train"]["train"]

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad")

tokenized_mrqa = mrqa.map(preprocess_training_examples, batched=True, 
                          remove_columns=mrqa["train"].column_names,
                          fn_kwargs={"tokenizer": tokenizer})

tokenized_mrqa.set_format(type="torch")

# Tokenizing evaluation dataset
tokenized_eval = mrqa["test"].map(preprocess_validation_examples, batched=True, 
                                  remove_columns=mrqa["test"].column_names,
                                  fn_kwargs={"tokenizer": tokenizer})
tokenized_eval.set_format(type="torch")

# Defining data collator
data_collator = DefaultDataCollator()

# Configuring parameters for the quantation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)

# Loading baseline model: DistilBert finetuned on Squadn dataset
model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad",
                                                       quantization_config=bnb_config,
                                                       device_map={"": 0})

# Calculating evaluation metrics before further finetuning
pre_training_metrics = eval_function(tokenized_eval, model, mrqa["test"])
print(f"Exact match before finetuning: {pre_training_metrics['exact_match']}\nF1 score before finetuning: {pre_training_metrics['f1']}, BLEU score before finetuning: {pre_training_metrics['bleu']}")

# Defining training parameters
output_dir_name = "trial_run_local"

# Configuring parameters of the low-rank adaptation and which layers of the network should be trained
peft_config = LoraConfig(
    lora_alpha=6,
    lora_dropout=0.15,
    r=4,
    bias="none",
    task_type="QUESTION_ANS",
    target_modules=["q_lin", "k_lin", "v_lin", "ffn.lin1", "ffn.lin2", "attention.out_proj"])

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir_name,
    eval_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='new_dir',
    load_best_model_at_end=True,
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

# Evaluating the model after the training and logging the evaluated results
post_training_metrics = eval_function(tokenized_eval, model, mrqa["test"])
best_ckpt_path = trainer.state.best_model_checkpoint

eval_logs = {
    "pre": pre_training_metrics,
    "post": post_training_metrics,
    "best model": best_ckpt_path
}

with open("eval_logs.json", "w") as file:
    json.dump(eval_logs, file)

print(f"Exact match after finetuning: {post_training_metrics['exact_match']}\nF1 score after finetuning: {post_training_metrics['f1']}")


