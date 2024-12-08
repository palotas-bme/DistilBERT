import os
import torch
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
import json
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from training_utils import *
from datasets import load_dataset
from datetime import datetime
import shutil

MRQA_SPLIT = 0.1
BEST_MODELS_DIR = "best models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training logs
with open("eval_logs.json", "r") as file:
    eval_logs = json.load(file)

# Load best mopdel checkpoint
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad")
base_model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased-distilled-squad").to(device)
model = PeftModel.from_pretrained(base_model, eval_logs["best model"]).to(device)

for root, dirs, files in os.walk("trial_run_local"):
    for dir in dirs:
        current_dir = os.path.join(root, dir)
        if not current_dir == os.path.normpath((eval_logs["best model"])):
             shutil.rmtree(current_dir)
             pass
        else:
            print(current_dir)

# Load the dataset
mrqa = load_dataset("mrqa", split="validation")

# Shuffle deterministically (optional) - so we always load the same split for evaluation
mrqa_shuffled = mrqa.shuffle(seed=42)

# Select mrqa subset
subset_size = int(MRQA_SPLIT * len(mrqa_shuffled))
mrqa_subset = mrqa_shuffled.select(range(subset_size))

# Tokenizing evaluation set
tokenized_mrqa = mrqa_subset.map(preprocess_validation_examples, batched=True, 
                                  remove_columns=mrqa.column_names,
                                  fn_kwargs={"tokenizer": tokenizer})

# Calculating eval metrics (same for all model trainings, helps compare models before deployment)
eval_metrics = eval_function(tokenized_mrqa, model, mrqa_subset)# Metrics to save
keys_to_save = ["exact_match", "f1", "bleu"]
metrics_to_save = {key: eval_metrics[key] for key in keys_to_save}

# File name for saving metrics
file_name = "model_metrics.json"

# Get the current date and time as a string
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(BEST_MODELS_DIR):
    os.mkdir(BEST_MODELS_DIR)

# Check if the file exists
if os.path.exists(os.path.join(BEST_MODELS_DIR, file_name)):
    # Load existing data and append new metrics
    with open(os.path.join(BEST_MODELS_DIR, file_name), "r") as file:
        data = json.load(file)
    data[current_time] = metrics_to_save
else:
    # Create a new dictionary with the current metrics
    data = {current_time: metrics_to_save}

# Save the updated data back to the file
with open(os.path.join(BEST_MODELS_DIR, file_name), "w") as file:
    json.dump(data, file, indent=4)

print(f"Metrics saved to '{file_name}' with timestamp '{current_time}'.")

# Moving model to best models directory
shutil.move(f"{os.path.normpath((eval_logs['best model']))}", 
            os.path.join(BEST_MODELS_DIR, f"{current_time}_best_model"))