#!/usr/bin/env python
# coding: utf-8

# In[26]:


# pip install pytesseract pillow pdf2image

import pytesseract
from pdf2image import convert_from_path

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Your paths ---
pdf_path = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\Answer pdf\2020Answer.pdf"
output_text_path = r"C:\Users\User\OneDrive\Desktop\Git Hub\ml-aqg-srilanka-ol-science\ML-for-Exams\data\AnswerText\2020Answer.txt"

# --- Poppler bin path ---
poppler_path = r"C:\poppler\poppler-24.07.0\Library\bin"   # <-- Change this to your actual poppler path

print("Converting PDF pages to images...")

# Convert PDF → Images
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

print("Running OCR on each page...")

full_text = ""

# OCR on each page
for i, img in enumerate(pages):
    print(f"OCR on page {i+1}...")
    text = pytesseract.image_to_string(img)
    full_text += text + "\n"

# Save to text file
with open(output_text_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("✅ OCR Complete! Text saved to:", output_text_path)

