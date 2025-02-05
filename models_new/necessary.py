import torch
from transformers import BertTokenizer, BertForSequenceClassification
import PyPDF2

def extract_details_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

    sections = {
        "Diagnosis Details": ["Diagnosis Details", "Diagnosis"],
        "Treatment Summary": ["Treatment Summary", "Treatment Details", "Treatment Report"],
        "Billing Summary": ["Billing Summary", "Financial Summary", "Billing Details"],
        "Pharmacy Bill": ["Pharmacy Bill", "Pharmacy Charges", "Medication Charges"],
    }

    section_dict = {key: "" for key in sections.keys()}
    current_section = None
    lines = text.splitlines()

    for line in lines:
        line = line.strip()  
        if not line:
            continue  

        found_section = False
        for section, variations in sections.items():
            if any(variation.lower() in line.lower() for variation in variations):
                current_section = section
                found_section = True
                break

        if current_section and not found_section:
            section_dict[current_section] += line + "\n"
            
    section_dict = {k: v.strip() for k, v in section_dict.items() if v.strip()}

    return section_dict

pdf_path = r'C:\Users\Harish bhalaa\Dropbox\PC\Desktop\CTS_PROJECT\Report\patient_report\Kidney_Failure.pdf'  # Replace with the path to your PDF file
extracted_sections = extract_details_from_pdf(pdf_path)
diagnosis_details = extracted_sections.get("Diagnosis Details", "")
treatment_summary = extracted_sections.get("Treatment Summary", "")

model = BertForSequenceClassification.from_pretrained('./necessary')
tokenizer = BertTokenizer.from_pretrained('./necessary')

device = torch.device("cpu")
model.to(device)

text = diagnosis_details + treatment_summary

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**inputs)

# Get the predicted class
prediction = outputs.logits.argmax(-1).item()

# Print the result
if prediction == 0:
    print("Mentioned treatment is neccessary for the diagnosis")
else:
    print("Mentioned treatment is experimental for the diagnosis")
