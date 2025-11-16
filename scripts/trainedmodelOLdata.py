#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install transformers datasets accelerate sentencepiece')


# In[7]:


pip install --upgrade transformers


# In[8]:


import pandas as pd
import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

# ----------------------------
# 1. LOAD DATASET
# ----------------------------
csv_path = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\OL_SciQ_dataset.csv"
df = pd.read_csv(csv_path)

# ----------------------------
# 2. CREATE MODEL INPUT + TARGET TEXT
# ----------------------------
df["input_text"] = (
    "Generate a multiple-choice question.\n"
    "Question: " + df["question"] + "\n"
    "Option A: " + df["distractor1"] + "\n"
    "Option B: " + df["distractor2"] + "\n"
    "Option C: " + df["distractor3"] + "\n"
    "Option D: " + df["correct_answer"]
)

df["target_text"] = df["correct_answer"]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Train/Validation Split
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ----------------------------
# 3. LOAD LOCAL MODEL
# ----------------------------
model_name = r"t5_sciq_local_model"  # Your local model folder

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ----------------------------
# 4. TOKENIZATION FUNCTION (BATCHEd + SAFE)
# ----------------------------
def preprocess(batch):
    # Convert Arrow arrays to Python lists of strings
    input_texts = [str(text) for text in batch["input_text"]]
    target_texts = [str(text) for text in batch["target_text"]]

    # Tokenize inputs
    model_input = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Tokenize labels
    labels = tokenizer(
        target_texts,
        padding="max_length",
        truncation=True,
        max_length=64
    )

    # Replace padding token id's in labels with -100 so they are ignored in loss
    labels_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    model_input["labels"] = labels_ids

    return model_input

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

# ----------------------------
# 5. TRAINING ARGUMENTS
# ----------------------------
training_args = TrainingArguments(
    output_dir="./t5_sciq_finetuned_output",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=False,
    report_to="none"
)

# ----------------------------
# 6. TRAINER
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# ----------------------------
# 7. TRAIN
# ----------------------------
trainer.train()

# ----------------------------
# 8. SAVE TRAINED MODEL
# ----------------------------
save_path = "t5_sri_lanka_ol_trained_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("Model saved to:", save_path)

