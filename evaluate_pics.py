import os
import cv2
import argparse
import easyocr
import numpy as np
from tqdm import tqdm

def get_variants(img):
    """
    Generate multiple image variants by splitting into color channels (RGB)
    and YUV components (Y, U, V), then creating grayscale-like versions for each channel.
    This helps OCR by giving it different visual perspectives of the text.
    """
    b, g, r = cv2.split(img)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return [
        (img, "Original"),
        (cv2.merge([r, r, r]), "R Channel"),
        (cv2.merge([g, g, g]), "G Channel"),
        (cv2.merge([b, b, b]), "B Channel"),
        (cv2.merge([y, y, y]), "Y Channel"),
        (cv2.merge([u, u, u]), "U Channel"),
        (cv2.merge([v, v, v]), "V Channel")
    ]

def extract_text(reader, variants):
    """
    Given a list of image variants, run OCR on each, and return the one with the highest average OCR confidence score.
    If no text is found, return "(no text)" with score 0.0.
    """
    best_score = 0.0
    best_text = "(no text)"
    for image, _ in variants:
        results = reader.readtext(image)
        if results:
            avg_score = sum([r[2] for r in results]) / len(results)
            if avg_score > best_score:
                best_score = avg_score
                best_text = " ".join([r[1] for r in results])
    return best_text, best_score

def evaluate_folder(input_folder, reader, mode="none"):
    """
    Evaluates each image in a folder using the specified OCR mode:
    - "none": raw image only
    - "flip": horizontally flipped image only
    - "full": original and flipped images with multiple variants

    Returns:
    - Average OCR confidence score across images that produce any text
    - Count of images that produce no text ("(no text)")
    - Total number of images (for percentage calculation)
    """
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    total_score = 0.0
    count_with_text = 0
    no_text_count = 0

    for fname in tqdm(image_files, desc=f"Evaluating ({mode})"):
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            no_text_count += 1
            continue

        if mode == "none":
            variants = [(img, "Raw")]
        elif mode == "flip":
            flipped = cv2.flip(img, 1)
            variants = [(img, "Raw"), (flipped, "Flipped")]
        elif mode == "full":
            variants = get_variants(img) + get_variants(cv2.flip(img, 1))
        else:
            raise ValueError("Mode must be 'none', 'flip', or 'full'")

        text, score = extract_text(reader, variants)
        if text == "(no text)":
            no_text_count += 1
        else:
            total_score += score
            count_with_text += 1

    total_images = len(image_files)
    avg_score = total_score / count_with_text if count_with_text > 0 else 0.0
    return avg_score, no_text_count, total_images

def main(input_folder):
    """
    Load the OCR reader and run three evaluations:
    - Without any transformation ("none")
    - With flipped images only ("flip")
    - With full variant processing ("full")
    
    Print the results for each, including percentage of images with no detected text.
    """
    reader = easyocr.Reader(['en'])

    avg_none, no_text_none, total_none = evaluate_folder(input_folder, reader, mode="none")
    avg_flip, no_text_flip, total_flip = evaluate_folder(input_folder, reader, mode="flip")
    avg_full, no_text_full, total_full = evaluate_folder(input_folder, reader, mode="full")

    print("\n--- Results ---")
    print(f"[Raw Image   ] Avg Score: {avg_none:.3f}, No Text Count: {no_text_none} ({(no_text_none / total_none) * 100:.2f}%)")
    print(f"[Flipped Only] Avg Score: {avg_flip:.3f}, No Text Count: {no_text_flip} ({(no_text_flip / total_flip) * 100:.2f}%)")
    print(f"[Full Variant] Avg Score: {avg_full:.3f}, No Text Count: {no_text_full} ({(no_text_full / total_full) * 100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR scoring on image folder (none, flip, full)")
    parser.add_argument("input_folder", type=str, help="Path to folder containing images")
    args = parser.parse_args()
    main(args.input_folder)