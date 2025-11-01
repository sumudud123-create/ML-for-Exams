# scripts/data_preprocessing.py
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)       # collapse whitespace
    text = re.sub(r'[^A-Za-z0-9.,?\'"() -]', '', text)  # remove odd chars
    return text.strip()

def main():
    input_path = "data/sample-questions.csv"
    output_path = "data/cleaned_sample_questions.csv"

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    # Apply simple cleaning to question_text and options
    df["question_text_clean"] = df["question_text"].apply(clean_text)
    for col in ["option_A","option_B","option_C","option_D"]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(clean_text)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
