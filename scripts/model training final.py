#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')


# In[5]:


import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# In[6]:


get_ipython().system('pip install transformers datasets sentencepiece accelerate evaluate sacrebleu nltk')


# In[7]:


import transformers
print("Transformers version:", transformers.__version__)


# In[8]:


import torch
import transformers
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast
print("✅ All core ML libraries are ready.")


# In[9]:


import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import evaluate
import random
import numpy as np


# In[10]:


# Path to your local SciQ CSV
csv_path = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\sciq_train.csv"

# Load using pandas
sciq_df = pd.read_csv(csv_path)

print("✅ Loaded dataset with", len(sciq_df), "rows")
print("Columns:", list(sciq_df.columns))
sciq_df.head()


# In[11]:


def make_pair(row):
    context = row.get("support", "")
    answer = row.get("correct_answer", "")
    input_text = f"generate_question: context: {context} answer: {answer}"
    target_text = str(row.get("question", ""))
    return {"input_text": input_text, "target_text": target_text}

pairs = [make_pair(r) for _, r in sciq_df.iterrows()]
pairs_df = pd.DataFrame(pairs)
pairs_df.head()


# In[12]:


from datasets import Dataset

# Shuffle and split (90% train, 10% eval)
pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.9 * len(pairs_df))
train_df = pairs_df.iloc[:split_idx]
eval_df = pairs_df.iloc[split_idx:]

train_ds = Dataset.from_pandas(train_df)
eval_ds = Dataset.from_pandas(eval_df)

print("Train:", len(train_ds), "Eval:", len(eval_ds))


# In[13]:


model_name = "t5-small"  # lightweight model
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# In[14]:


MAX_INPUT = 256
MAX_TARGET = 64

def preprocess(examples):
    inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=MAX_INPUT)
    targets = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=MAX_TARGET)
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in targets["input_ids"]
    ]
    return inputs

train_tokenized = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_tokenized = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)


# In[15]:


get_ipython().system('pip install -U transformers')


# In[16]:


from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq


# In[17]:


from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="t5_sciq_local_qg",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    do_eval=True,              # use this instead of evaluation_strategy
    logging_steps=200,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[18]:


sacrebleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode([[l for l in label if l != -100] for label in labels], skip_special_tokens=True)
    bleu = sacrebleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])
    return {"bleu": bleu["score"]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[19]:


def generate_question(support, answer):
    input_text = f"generate_question: context: {support} answer: {answer}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sample = sciq_df.sample(1).iloc[0]
print("Context:", sample["support"])
print("Answer:", sample["correct_answer"])
print("Generated Question:", generate_question(sample["support"], sample["correct_answer"]))
print("Actual Question:", sample["question"])



# In[20]:


trainer.train()


# In[21]:


def generate_question(support, answer):
    input_text = f"generate_question: context: {support} answer: {answer}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

sample = sciq_df.sample(1).iloc[0]
print("Context:", sample["support"])
print("Answer:", sample["correct_answer"])
print("Generated Question:", generate_question(sample["support"], sample["correct_answer"]))
print("Actual Question:", sample["question"])


# In[22]:


model.save_pretrained("t5_sciq_local_model")
tokenizer.save_pretrained("t5_sciq_local_model")
print("✅ Model saved successfully.")

