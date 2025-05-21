import os
import torch
import numpy
import sklearn
import matplotlib
import easyocr
import re
from rapidfuzz import fuzz
import csv

# ----------------------------
# Data loading
# ----------------------------

def load_ocr_results_with_filenames(filepath):
    """
    Load OCR results from a text file with the format: filename|||text|||score
    Returns a dictionary mapping filenames to (text, score) tuples.
    """
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|||")
            if len(parts) == 3:
                fname, text, score = parts
                results[fname] = (text, float(score))
    return results

# ----------------------------
# Text cleaning
# ----------------------------

def clean_text(text):
    """
    Clean and normalize OCR text by:
    - Lowercasing
    - Replacing digits with similar-looking letters
    - Removing punctuation
    - Collapsing multiple spaces into one
    Returns the cleaned text string.
    """
    text = text.lower()

    num_to_char = {
        '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
        '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g'
    }
    for num, char in num_to_char.items():
        text = text.replace(num, char)

    text = re.sub(r'[^\w\s]', ' ', text)         # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # collapse multiple spaces
    return text

# ----------------------------
# Filter by OCR confidence
# ----------------------------

def filter_by_score(ocr_dict, threshold):
    """
    Filter OCR results based on confidence score.
    Returns a dictionary with only entries >= threshold.
    """
    return {
        fname: (text, score)
        for fname, (text, score) in ocr_dict.items()
        if score >= threshold
    }

# ----------------------------
# Clean a full OCR result set
# ----------------------------

def clean_ocr_result_dict(ocr_dict):
    """
    Apply clean_text() to each text in an OCR result dictionary.
    Returns a new dictionary with cleaned texts.
    """
    return {
        fname: (clean_text(text), score)
        for fname, (text, score) in ocr_dict.items()
    }

# ----------------------------
# Save to CSV
# ----------------------------

def save_results_to_csv(csv_name, alcohol_cleaned_results, non_alcohol_cleaned_results):
    """
    Save cleaned alcohol and non-alcohol OCR results into a CSV file
    with columns: 'text' and 'prediction'.
    """
    csv_data = []

    for fname, (text, score) in alcohol_cleaned_results.items():
        csv_data.append({"text": text, "prediction": "alcohol"})

    for fname, (text, score) in non_alcohol_cleaned_results.items():
        csv_data.append({"text": text, "prediction": "non_alcohol"})

    with open(f"{csv_name}.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["text", "prediction"])
        writer.writeheader()
        writer.writerows(csv_data)

# ----------------------------
# Main Processing
# ----------------------------

# Load OCR results from files
alcohol_ocr_dict = load_ocr_results_with_filenames("images/alcohol/ocr_results.txt")
non_alcohol_ocr_dict = load_ocr_results_with_filenames("images/non_alcohol/ocr_results.txt")

# Filter by OCR confidence score
threshold = 0.26
alcohol_filtered_results = filter_by_score(alcohol_ocr_dict, threshold)
non_alcohol_filtered_results = filter_by_score(non_alcohol_ocr_dict, threshold)

# Clean the text
alcohol_cleaned_results = clean_ocr_result_dict(alcohol_filtered_results)
non_alcohol_cleaned_results = clean_ocr_result_dict(non_alcohol_filtered_results)

# Save final cleaned dataset
save_results_to_csv("ocr_text_classification", alcohol_cleaned_results, non_alcohol_cleaned_results)
