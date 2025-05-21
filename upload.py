from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi

# Replace with your HF username and desired model repo name
repo_name = "alcohol_net"
hf_username = "NoamDiamant52"  # change this

# Load the model & tokenizer from your saved directory
model = AutoModelForSequenceClassification.from_pretrained("alcohol_net",  num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Or "alcohol_net" if you saved tokenizer too

# Push to hub
model.push_to_hub(f"{hf_username}/{repo_name}")
tokenizer.push_to_hub(f"{hf_username}/{repo_name}")
