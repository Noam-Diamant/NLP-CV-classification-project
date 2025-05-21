import os, cv2, torch, argparse, easyocr, re
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rapidfuzz import fuzz

# ----------------------------
# Define keyword lists
# ----------------------------

ALCOHOL_KEYWORDS = [
    'vodka', 'rum', 'whiskey', 'bourbon', 'tequila', 'cocktail', 'gin', 'smirnoff', 'guinness',
    'absolut', 'heineken', 'brandy', 'stellaartois', 'wine', 'beer', 'lager', 'alcohol',
    'champagne', 'johnniewalker', 'budweiser', 'jackdaniels', 'chardonnay', 'port', 'merlot'
]

NON_ALCOHOL_KEYWORDS = [
    'chocolate', 'hot', 'iced', 'juice', 'espresso', 'cbd', 'coffee', 'herbaltea', 'soda',
    'cannabis', 'weed', 'marijuana', 'sparklingwater', 'sparkling', 'lemonade', 'matcha',
    'icetea', 'icedtea', 'tea', 'water', 'milk', 'milkshake'
]

# ----------------------------
# Utility Functions
# ----------------------------

def clean_text(text):
    """
    Lowercase text, replace digits with letters, remove punctuation, and reduce whitespace.
    """
    text = text.lower()
    num_to_char = {'0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
                   '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g'}
    for num, char in num_to_char.items():
        text = text.replace(num, char)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fuzzy_contains_keyword(text, keywords, threshold=82):
    """
    Check if the text approximately contains any keyword from a list.
    """
    joined = text.replace(" ", "")
    for keyword in keywords:
        if fuzz.partial_ratio(joined, keyword) >= threshold:
            return True
    return False

def get_variants(img):
    """
    Generate variants of the image using color channels and YUV decomposition.
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

def extract_text_from_image(image_path, reader):
    """
    Runs OCR on various visual variants of the image and returns the best-scoring text result.
    """
    image = cv2.imread(image_path)
    if image is None:
        return "(no text)", 0.0
    variants = get_variants(image)
    flipped = cv2.flip(image, 1)
    variants += get_variants(flipped)

    best_text = "(no text)"
    best_score = 0.0
    for variant_img, _ in variants:
        results_raw = reader.readtext(variant_img)
        if results_raw:
            avg_score = sum(r[2] for r in results_raw) / len(results_raw)
            text = " ".join([r[1] for r in results_raw])
            if avg_score > best_score:
                best_score = avg_score
                best_text = text
    return best_text, best_score

def hybrid_predict(text, nn_pred):
    """
    Apply fuzzy matching logic first. If ambiguous, fallback to neural network prediction.
    """
    is_alcohol = fuzzy_contains_keyword(text, ALCOHOL_KEYWORDS)
    is_non_alcohol = fuzzy_contains_keyword(text, NON_ALCOHOL_KEYWORDS)
    if is_alcohol and not is_non_alcohol:
        return "alcohol"
    elif is_non_alcohol and not is_alcohol:
        return "non_alcohol"
    else:
        return "alcohol" if nn_pred == 0 else "non_alcohol"

# ----------------------------
# Main Runner
# ----------------------------

def main(input_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = easyocr.Reader(['en'])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("NoamDiamant52/alcohol_net")
    model.to(device)
    model.eval()

    results = []
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in tqdm(image_files, desc="Classifying images"):
        full_path = os.path.join(input_folder, fname)

        # OCR + clean text
        raw_text, _ = extract_text_from_image(full_path, reader)
        cleaned_text = clean_text(raw_text)

        # NN prediction
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=16)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            nn_pred = torch.argmax(logits, dim=-1).item()

        # Hybrid decision
        prediction = hybrid_predict(cleaned_text, nn_pred)
        results.append({"file_name": fname, "prediction": prediction})

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

# ----------------------------
# Command Line Interface
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images as alcohol or non_alcohol")
    parser.add_argument("input_folder", type=str, help="Folder path with images to classify")
    args = parser.parse_args()
    main(args.input_folder)