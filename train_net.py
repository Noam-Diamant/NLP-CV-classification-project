# ----------------------------
# Imports
# ----------------------------
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
from sklearn.metrics import classification_report

# ----------------------------
# Configuration
# ----------------------------

data_path = "ocr_text_classification.csv"  # CSV containing text and labels
text_column_name = "text"                  # Column with the input text
label_column_name = "prediction"           # Column with raw labels (e.g., "alcohol", "non_alcohol")

model_name = "distilbert-base-uncased"     # Pretrained transformer model
num_labels = 2                             # Binary classification: 0 = alcohol, 1 = non_alcohol

# ----------------------------
# Load and encode data
# ----------------------------

df = pd.read_csv(data_path)                # Load data into a DataFrame

le = preprocessing.LabelEncoder()          # Create label encoder
le.fit(df[label_column_name].tolist())     # Fit on existing labels
df['label'] = le.transform(df[label_column_name].tolist())  # Create numerical labels

# ----------------------------
# Split data into train/val
# ----------------------------

targets_idx = np.arange(len(df))           # All row indices
labels = df['label']                       # Encoded label values

train_idx, val_idx = train_test_split(     # Stratified split to preserve class balance
    targets_idx,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=labels
)

train_df = df.iloc[train_idx]              # Training data
val_df = df.iloc[val_idx]                  # Validation data

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(val_df)

# ----------------------------
# Tokenization
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    """
    Tokenize and pad/truncate input text for transformer input.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=16,
        padding='max_length',
        return_attention_mask=True
    )

# Apply tokenization
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# ----------------------------
# Model setup
# ----------------------------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Handles dynamic padding

metric = evaluate.load("accuracy")  # Load accuracy metric

def compute_metrics(eval_pred):
    """
    Compute accuracy from predicted logits and ground truth labels.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ----------------------------
# Training setup
# ----------------------------

training_args = TrainingArguments(
    output_dir="./results",                # Save outputs here
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",           # Evaluate at the end of every epoch
    logging_strategy="epoch"               # Log metrics at each epoch
)

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
# Training
# ----------------------------

trainer.train()                            # Train the model

trainer.save_model('alcohol_net')         # Save the fine-tuned model

# ----------------------------
# Evaluation (Train set)
# ----------------------------

preds = trainer.predict(tokenized_train)
preds = np.argmax(preds[:3][0], axis=1)    # Extract predicted labels from logits
GT = train_df['label'].tolist()            # Ground truth labels
print(classification_report(GT, preds))    # Print precision/recall/f1

# ----------------------------
# Evaluation (Validation set)
# ----------------------------

preds = trainer.predict(tokenized_test)
preds = np.argmax(preds[:3][0], axis=1)
GT = val_df['label'].tolist()
print(classification_report(GT, preds))