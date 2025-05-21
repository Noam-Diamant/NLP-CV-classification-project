import os
import cv2
import easyocr
from tqdm import tqdm

def extract_best_texts_from_folder(folder_path, save_to_txt=True):
    """
    Perform OCR on all images in a folder using EasyOCR with multiple variants (e.g., flipped, YUV channels).
    Selects the variant with the highest average OCR confidence per image.
    Optionally saves results to a text file in the format: filename|||text|||score.
    Returns a dictionary mapping filenames to (best_text, best_score).
    """
    reader = easyocr.Reader(['en'])  # Load EasyOCR model for English
    results = {}

    # Get list of image files in the folder (PNG, JPG, JPEG)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image file
    for filename in tqdm(image_files, desc="Processing images"):
        full_path = os.path.join(folder_path, filename)
        image = cv2.imread(full_path)  # Load image
        if image is None:
            continue  # Skip unreadable or corrupted images

        def get_variants(img):
            """
            Generate different visual variants of the input image,
            including color channel isolations and YUV splits.
            Returns a list of (variant_image, label) tuples.
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

        # Create variants of the image (original + flipped)
        variants = get_variants(image)
        flipped = cv2.flip(image, 1)            # Horizontal flip
        variants += get_variants(flipped)       # Add flipped variants too

        best_text = "(no text)"  # Default value
        best_score = 0.0         # Track best confidence

        # Evaluate OCR on each image variant
        for variant_img, label in variants:
            results_raw = reader.readtext(variant_img)  # Run OCR
            if results_raw:
                avg_score = sum(r[2] for r in results_raw) / len(results_raw)  # Mean confidence
                text = " ".join([r[1] for r in results_raw])                   # Join all text fragments
                if avg_score > best_score:  # Update if this variant is better
                    best_score = avg_score
                    best_text = text

        # Save the best text and score for the image
        results[filename] = (best_text, best_score)

    # Optionally save results to file
    if save_to_txt:
        output_file = os.path.join(folder_path, f"ocr_results.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for fname, (text, score) in results.items():
                f.write(f"{fname}|||{text}|||{score:.4f}\n")
        print(f"Results saved to {output_file}")

    return results

def load_ocr_results_with_filenames(filepath):
    """
    Load OCR results from a text file with format: filename|||text|||score
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
# Run OCR extraction for both folders
# ----------------------------

# Apply OCR extraction and save results for alcohol images
alcohol_results = extract_best_texts_from_folder("images/alcohol")

# Apply OCR extraction and save results for non-alcohol images
non_alcohol_results = extract_best_texts_from_folder("images/non_alcohol")
