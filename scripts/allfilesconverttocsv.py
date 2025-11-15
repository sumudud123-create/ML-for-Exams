#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import csv

# 📌 Path to your folder containing ALL edited text files
input_folder = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\PaperText"

# 📌 Path to final SciQ CSV output
output_csv = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\OL_SciQ_dataset.csv"

dataset = []

# 📌 Regex pattern matching your exact question layout
pattern = re.compile(
    r'\d+\.\s*(.*?)\s*'                   # question text
    r'\(1\)\s*([^\(]*?)\s*'               # distractor1
    r'\(2\)\s*([^\(]*?)\s*'               # distractor2
    r'\(3\)\s*([^\(]*?)\s*'               # distractor3
    r'\(4\)\s*([^\n]*?)\s*'               # correct answer
    r'(?=\n\d+\.|\Z)',                    # next question or end of text
    re.DOTALL
)

print("Extracting questions...")

# 📌 Process every text file in folder
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        filepath = os.path.join(input_folder, file)
        print("Processing:", file)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        matches = pattern.findall(text)

        print(f"  Found {len(matches)} questions.")

        for question, d1, d2, d3, correct in matches:
            dataset.append({
                "question": question.strip(),
                "distractor1": d1.strip(),
                "distractor2": d2.strip(),
                "distractor3": d3.strip(),
                "correct_answer": correct.strip(),
                "support": ""
            })

# 📌 Export to CSV
print("\nSaving SciQ dataset...")

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "question",
        "distractor1",
        "distractor2",
        "distractor3",
        "correct_answer",
        "support"
    ])
    writer.writeheader()
    writer.writerows(dataset)

print("\n🎉 DONE!")
print("Total questions saved:", len(dataset))
print("Final SciQ CSV saved to:")
print(output_csv)

