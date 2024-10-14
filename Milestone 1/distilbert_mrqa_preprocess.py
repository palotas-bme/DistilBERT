from datasets import load_dataset
from transformers import AutoTokenizer
import json

# The MRQA dataset is included in huggingface's datasets library, so we just have to load it
mrqa = load_dataset("mrqa", split="train[:20%]")
mrqa = mrqa.train_test_split(test_size=0.2)
mrqa["train"] = mrqa["train"].train_test_split(test_size=0.2)
mrqa["val"] = mrqa["train"]["test"]
mrqa["train"] = mrqa["train"]["train"]


# Even though tha dataset contains the tokenized versions of both questions and contexts, it is better to use the distilbert's own pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Tokenizing the texts with distilbert's own pretrained tokenizer and mapping the answer start and end character indices onto the tokenized text
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    # Tokenizing inputs
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["detected_answers"]
    start_positions = []
    end_positions = []

    # Mapping the answer start and end characters to the tokenized text
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
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

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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

tokenized_mrqa = mrqa.map(preprocess_function, batched=True, remove_columns=mrqa["train"].column_names)

# Saving the tokenized contexts and answers to json files
train = tokenized_mrqa["train"].to_dict()
test = tokenized_mrqa["test"].to_dict()
val = tokenized_mrqa["val"].to_dict()


with open("train.json", "w") as f:
    json.dump(train, f)
with open("test.json", "w") as f:
    json.dump(test, f)
with open("val.json", "w") as f:
    json.dump(val, f)