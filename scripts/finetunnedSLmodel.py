#!/usr/bin/env python
# coding: utf-8

# In[10]:


from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd

# Load dataset
csv_path = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\OL_SciQ_dataset.csv"
df = pd.read_csv(csv_path)

# Update target_text as above
df["input_text"] = (
    "Generate a multiple-choice question.\n"
    "Question: " + df["question"] + "\n"
    "Option A: " + df["distractor1"] + "\n"
    "Option B: " + df["distractor2"] + "\n"
    "Option C: " + df["distractor3"] + "\n"
    "Option D: " + df["correct_answer"]
)
df["target_text"] = (
    "Question: " + df["question"] + "\n"
    "A: " + df["distractor1"] + "\n"
    "B: " + df["distractor2"] + "\n"
    "C: " + df["distractor3"] + "\n"
    "D: " + df["correct_answer"]
)

dataset = Dataset.from_pandas(df)
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load base T5
model_name = "t5_sciq_local_model"  # or "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess(batch):
    input_texts = [str(t) for t in batch["input_text"]]
    target_texts = [str(t) for t in batch["target_text"]]
    model_input = tokenizer(input_texts, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(target_texts, padding="max_length", truncation=True, max_length=256)
    labels_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in seq] for seq in labels["input_ids"]]
    model_input["labels"] = labels_ids
    return model_input

train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5_mcq_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train model
trainer.train()

# Save model
trainer.save_model("./t5_mcq_finetuned")
tokenizer.save_pretrained("./t5_mcq_finetuned")

