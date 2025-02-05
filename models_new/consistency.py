from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the saved model and tokenizer
model_name = "../models_new/consistency"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def preprocess_text(text):
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return encoding

# Example text
text = """The patient was diagnosed with a brain tumor after undergoing a series of diagnostic tests
including an MRI and a biopsy. The scans revealed there is no significant mass located in the
frontal lobe, which is not contributing any symptoms and does not pose a substantial risk to
the patient's health."""

# Preprocess text
inputs = preprocess_text(text)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities or class labels
predictions = torch.argmax(logits, dim=-1)
label = predictions.item()

# Convert label to human-readable form
labels = ["inconsistent", "consistent"]
result = labels[label]

print(f"Prediction: {result}")








