from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
import evaluate
from sklearn.metrics import classification_report
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
# Fuzzy keyword detection
# ----------------------------

def fuzzy_contains_keyword(text, keywords, threshold=82):
    """
    Check if the input text contains any of the provided keywords using fuzzy partial ratio.
    The text is stripped of whitespace for better matching accuracy.
    Returns True if any keyword matches above the threshold.
    """
    joined = text.replace(" ", "")
    for keyword in keywords:
        score = fuzz.partial_ratio(joined, keyword)
        if score >= threshold:
            return True
    return False

# ----------------------------
# Hybrid classification logic
# ----------------------------

def hybrid_predict(texts, nn_predictions):
    """
    For each text, if fuzzy match clearly indicates alcohol or non-alcohol (but not both),
    return the corresponding class (0 or 1). Otherwise, fallback to neural network prediction.
    """
    hybrid_preds = []
    for text, nn_pred in zip(texts, nn_predictions):
        is_alcohol = fuzzy_contains_keyword(text, ALCOHOL_KEYWORDS)
        is_non_alcohol = fuzzy_contains_keyword(text, NON_ALCOHOL_KEYWORDS)
        if is_alcohol and not is_non_alcohol:
            hybrid_preds.append(0)  # Alcohol
        elif is_non_alcohol and not is_alcohol:
            hybrid_preds.append(1)  # Non-Alcohol
        else:
            hybrid_preds.append(nn_pred)  # In the ambiguous case - use NN
    return hybrid_preds

# ----------------------------
# Tokenization preprocessing
# ----------------------------

def preprocess_function(examples):
    """
    Tokenizes each example's 'text' using a fixed max length and padding.
    Used for preparing inputs to the transformer model.
    """
    return tokenizer(examples["text"], truncation=True, max_length=16, padding='max_length', return_attention_mask=True)

# ----------------------------
# Accuracy metric computation
# ----------------------------

def compute_metrics(eval_pred):
    """
    Extracts logits and labels from the evaluation predictions and computes accuracy.
    Used during Trainer evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ----------------------------
# Setup and Initialization
# ----------------------------

data_path = "ocr_text_classification.csv"
text_column_name = "text"
label_column_name = "prediction"

model_name = "distilbert-base-uncased"
num_labels = 2

# Load the fine-tuned model and tokenizer
hf_username = "NoamDiamant52"
model = AutoModelForSequenceClassification.from_pretrained(f"{hf_username}/alcohol_net", num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and encode dataset labels
df = pd.read_csv(data_path)
le = preprocessing.LabelEncoder()
le.fit(df[label_column_name].tolist())
df['label'] = le.transform(df[label_column_name].tolist())

# ----------------------------
# Train/Test Split
# ----------------------------

targets_idx = np.arange(len(df))
labels = df['label']

train_idx, val_idx = train_test_split(
    targets_idx,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=labels
)

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(val_df)

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Padding and metric setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

# Setup Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ----------------------------
# Predictions (NN and Hybrid)
# ----------------------------

# Get NN predictions
nn_train_preds = trainer.predict(tokenized_train).predictions
nn_train_preds = np.argmax(nn_train_preds, axis=1)

nn_test_preds = trainer.predict(tokenized_test).predictions
nn_test_preds = np.argmax(nn_test_preds, axis=1)

# Compute hybrid predictions (fuzzy+NN fallback)
hybrid_train_preds = hybrid_predict(train_df['text'].tolist(), nn_train_preds)
hybrid_test_preds = hybrid_predict(val_df['text'].tolist(), nn_test_preds)

# ----------------------------
# Evaluation Reports
# ----------------------------

print("====== NEURAL NETWORK ONLY (TRAIN) ======")
print(classification_report(train_df['label'].tolist(), nn_train_preds, target_names=le.classes_))

print("\n====== HYBRID CLASSIFIER (TRAIN) ======")
print(classification_report(train_df['label'].tolist(), hybrid_train_preds, target_names=le.classes_))

print("\n====== NEURAL NETWORK ONLY (VAL) ======")
print(classification_report(val_df['label'].tolist(), nn_test_preds, target_names=le.classes_))

print("\n====== HYBRID CLASSIFIER (VAL) ======")
print(classification_report(val_df['label'].tolist(), hybrid_test_preds, target_names=le.classes_))