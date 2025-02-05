from flask import Flask, request, redirect, url_for, render_template, flash
from pymongo import MongoClient
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import re
from docx import Document
import pandas as pd
from fpdf import FPDF
import google.generativeai as genai
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'harish_secret_key'  

client = MongoClient('mongodb+srv://harishbhalaa:harish07@cluster0.mvfsr.mongodb.net/claims_processing_assistant',
                     tls=True, tlsAllowInvalidCertificates=True)
db = client['claims_processing_assistant']  
patients_collection = db['patient']
policies_collection = db['policies']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pc = Pinecone(api_key="22c0c522-9df5-4e58-9816-6d004936cc21")
index_name = "icd"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric='cosine',  
        spec=ServerlessSpec(
            cloud='aws',  
            region='us-east-1'  
        )
    )
index = pc.Index(index_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

valid_specializations = {
    "cardiac": ["Cardiol"],
    "lung": ["Pulmonologist"],
    "kidney": ["Nephrolog ist"],
    "brain": ["Neuro"],
    "spinal": ["Orthopedist"],
    "liver": ["Hepatologist"],
    "allergic": ["Allergist"],
    "eye": ["Ophthalmologist"],
    "cancer": ["Oncologist"],
    "urinary": ["Urologist"],
    "diabetes": ["Endocrinologist"],
}

tokenizer_cons = BertTokenizer.from_pretrained('./models/consistency')
model_cons = BertForSequenceClassification.from_pretrained('./models/consistency')
device = torch.device("cpu")
model_cons.to(device)

tokenizer_nec = BertTokenizer.from_pretrained('./models/necessary')
model_nec = BertForSequenceClassification.from_pretrained('./models/necessary')
device = torch.device("cpu")
model_nec.to(device)

genai.configure(api_key="AIzaSyApXTUP1odGSHWpG-EgfPLb9JyUnkIi_t4")
generation_config = {
  "temperature": 0.1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  system_instruction="i will give you patient report summary document extracted text which includes patient information, handled doctor information, diagnosis, treatment, hospital and pharmacy bills. i checked 6 criteria to approve or reject the claim. They are patient has policy or not, diagnosis of patient covered under the policy or not,date of admission is covered under the expiry date or not, ,doctor specialization is valid to diagnose the specific disease or not , diagnosis is  consistent or inconsistent (whether identifying the diagnosis is supported by the test and scanned results), the treatment is necessary or experimental. \n\nif patient has policy, policy covered under diagnosis, admission dates are under the expiration date, doctor specialization is valid, diagnosis is consistent, treatment is necessary. then approve the claim. if any of the condition is false then reject the claim. if the claim is rejected then don't mention the bill calulation( if any of the one condition not met means claim reject means don't mention bill calculation stop with rejection reason\n\nif the claim is approved then show the criteria which are checked to approve and  bills amount is mentioned in the input so you have to include the bills in the approval.\nif the claim is rejected then show the criteria which are checked to approve and provide the accurate reason for the rejection of claim should be mentioned properly. \n\nusing this you have to generate a summary report text which should be in the format\n\nformat of the summary document :\npatient information\ndoctor information\nhospital details \n\ndiagnosis summary with ICD code , \ntreatment plan \n\nElligibility check \nput in table to check the eligibility check \npatient has policy                            yes/ no  reason\ndiagonsis covered under policy      yes / no     reason\ndate of administartion                        yes/no      reason\nDoctor specialization                         yes/no     reason\nthe diagonsis  consistence              yes/no       reason\ntreament necessary                          yes /no       reason\n",
)
chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy\ndiagonsis covered under policy ,\ndate of administration is under the expiration (date in policy of patient\neffective_date - 2024-01-01\nexpiration_date - 2024-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05) ,\nDoctor specialization is valid ,\nthe diagonsis is consistence,\ntreament is necessary",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | Yes |  Lung cancer is covered under the policy. |\n| Date of administration | Yes |  Date of Admission (2024-09-05) falls within the policy coverage period (2024-01-01 to 2024-12-31). |\n| Doctor specialization | Yes | Pulmonologist is a valid specialization for diagnosing and treating lung cancer. |\n| The diagnosis is consistent | Yes | The diagnosis of lung cancer is supported by the CT scan and biopsy results. |\n| Treatment necessary | Yes | The recommended treatment plan is necessary and appropriate for managing the patient's advanced stage lung cancer. |\n\n**Claim Approved**\n\n**Bill Summary**\n\n**Hospital Main Bill**\n\n| Item | Amount (₹) |\n|---|---|\n| Room Charges (7 days) | ₹28000 |\n| ICU Charges (5 days) | ₹50000 |\n| Lung Cancer Surgery | ₹400000 |\n| Doctor's Fees | ₹30000 |\n| Nursing Charges | ₹20000 |\n| **Total Hospital Bill** | **₹528000** |\n\n**Pharmacy Bill**\n\n| Medication | Amount (₹) |\n|---|---|\n| Chemotherapy Drugs (30 days) | ₹60000 |\n| Pain Management | ₹7000 |\n| General Medication | ₹10000 |\n| **Total Pharmacy Bill** | **₹77000** |\n\n**Grand Total: ₹605000** \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has no policy\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | No |  The patient does not have a valid insurance policy. |\n| Diagnosis covered under policy | N/A |  Not applicable as the patient does not have a policy. |\n| Date of administration | N/A |  Not applicable as the patient does not have a policy. |\n| Doctor specialization | N/A |  Not applicable as the patient does not have a policy. |\n| The diagnosis is consistent | N/A |  Not applicable as the patient does not have a policy. |\n| Treatment necessary | N/A |  Not applicable as the patient does not have a policy. |\n\n**Claim Rejected** \n\n**Reason for Rejection:** The patient does not have a valid insurance policy to cover the medical expenses.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\n\npatient has policy\npatient diagonsis  not covered under policy ",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | No | Lung cancer is not covered under the patient's insurance policy. |\n| Date of administration | N/A | Not applicable as the diagnosis is not covered. |\n| Doctor specialization | N/A | Not applicable as the diagnosis is not covered. |\n| The diagnosis is consistent | N/A | Not applicable as the diagnosis is not covered. |\n| Treatment necessary | N/A | Not applicable as the diagnosis is not covered. |\n\n**Claim Rejected** \n\n**Reason for Rejection:** Lung cancer is not covered under the patient's insurance policy. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy,\ndiagonsis covered under policy ,\ndate of adminition is not under the expiration (date in policy of patient\neffective_date - 2023-01-01\nexpiration_date - 2023-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05)",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | Yes | Lung cancer is covered under the patient's insurance policy. |\n| Date of administration | No | The date of admission (2024-09-05) falls outside the policy coverage period (2023-01-01 to 2023-12-31). |\n| Doctor specialization | N/A |  Not applicable as the date of administration is not within the policy period. |\n| The diagnosis is consistent | N/A |  Not applicable as the date of administration is not within the policy period. |\n| Treatment necessary | N/A |  Not applicable as the date of administration is not within the policy period. |\n\n**Claim Rejected** \n\n**Reason for Rejection:** The patient's insurance policy expired on 2023-12-31, and the date of admission was 2024-09-05. Therefore, the claim falls outside the policy coverage period. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy,\ndiagonsis covered under policy ,\ndate of adminition is under the expiration (date in policy of patient\neffective_date - 2024-01-01\nexpiration_date - 2024-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05)\n\nDoctor specialization is not valid .\n",
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy,\ndiagonsis covered under policy ,\ndate of adminition is not under the expiration (date in policy of patient\neffective_date - 2024-01-01\nexpiration_date - 2024-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05)\n\nDoctor specialization is valid ,\nthe diagonsis is  not consistence.\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | Yes | Lung cancer is covered under the patient's insurance policy. |\n| Date of administration | Yes | The date of admission (2024-09-05) falls within the policy coverage period (2024-01-01 to 2024-12-31). |\n| Doctor specialization | No | Pulmonologist is not a valid specialization for diagnosing and treating lung cancer. This requires a specialist such as an oncologist or thoracic surgeon. |\n| The diagnosis is consistent | N/A |  Not applicable as the doctor specialization is not valid. |\n| Treatment necessary | N/A |  Not applicable as the doctor specialization is not valid. |\n\n**Claim Rejected** \n\n**Reason for Rejection:**  The doctor's specialization, Pulmonologist, is not valid for diagnosing and treating lung cancer. The claim requires a specialist such as an oncologist or thoracic surgeon. \n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy,\ndiagonsis covered under policy ,\ndate of adminition is  under the expiration (date in policy of patient\neffective_date - 2024-01-01\nexpiration_date - 2024-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05)\n\nDoctor specialization is valid ,\nthe diagonsis is  not consistence.\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | Yes | Lung cancer is covered under the patient's insurance policy. |\n| Date of administration | Yes | The date of admission (2024-09-05) falls within the policy coverage period (2024-01-01 to 2024-12-31). |\n| Doctor specialization | Yes | Pulmonologist is a valid specialization for diagnosing and treating lung cancer. |\n| The diagnosis is consistent | No | The diagnosis of lung cancer needs to be further investigated as it's not clear whether the CT scan and biopsy results fully support the diagnosis.  More information is needed to confirm the consistency of the diagnosis. |\n| Treatment necessary | N/A |  Not applicable as the diagnosis consistency is not confirmed. |\n\n**Claim Rejected** \n\n**Reason for Rejection:** The diagnosis of lung cancer requires further investigation to confirm its consistency with the available medical evidence (CT scan and biopsy results).  The claim cannot be approved until the diagnosis is confirmed. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Global Health Institute\nPatient Information\nName:\nSowmiya\nDate of Birth:\nMarch 20, 1965\nGender:\nFemale\nPatient ID:\nGHI123456\nInsurance Policy Number:\nINS123456\nDoctor Information\nName:\nJohn Doe\nSpecialization:\nPulmonologist\nDiagnosis Details\nThe patient was diagnosed with lung cancer following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage, indicating a severe health risk that requires immediate intervention to prevent further complications.\nTreatment Summary\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells. This treatment approach is essential to managing the patient's condition and is necessary to prevent the spread of the disease.\nBilling Summary\nHospital Main Bill\nAmount (₹)\nRoom Charges (7 days)\n₹28000\nICU Charges (5 days)\n₹50000\nLung Cancer Surgery\n₹400000\nDoctor's Fees\n₹30000\nNursing Charges\n₹20000\nPharmacy Bill\nMedication\nAmount (₹)\nChemotherapy Drugs (30 days)\n₹60000\nPain Management\n₹7000\nGeneral Medication\n₹10000\nTotal Amount Payable\nTotal Hospital Bill: ₹528000 Total Pharmacy Bill: ₹77000 Grand Total: ₹605000\npatient has policy,\ndiagonsis covered under policy ,\ndate of adminition is not under the expiration (date in policy of patient\neffective_date - 2024-01-01\nexpiration_date - 2024-12-31\ndate of admission\ndate of admission - 2024-09-05\ndate of discharge - 2024-09-05)\n\nDoctor specialization is valid ,\nthe diagonsis is  is consistence,\ntreament is  not necessary\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## Claim Summary Report\n\n**Patient Information**\nName: Sowmiya\nDate of Birth: March 20, 1965\nGender: Female\nPatient ID: GHI123456\nInsurance Policy Number: INS123456\n\n**Doctor Information**\nName: John Doe\nSpecialization: Pulmonologist\n\n**Hospital Details**\nGlobal Health Institute\n\n**Diagnosis Summary**\nThe patient was diagnosed with lung cancer (ICD Code: C34.9) following a series of diagnostic tests including a CT scan and a biopsy. The tests revealed a malignant tumor in the left lung, which has progressed to an advanced stage.\n\n**Treatment Plan**\nThe recommended treatment plan includes a combination of surgery to remove the tumor, followed by chemotherapy and radiation therapy to eliminate any remaining cancerous cells.\n\n**Eligibility Check**\n\n| Criteria | Result | Reason |\n|---|---|---|\n| Patient has policy | Yes | Policy Number: INS123456 |\n| Diagnosis covered under policy | Yes | Lung cancer is covered under the patient's insurance policy. |\n| Date of administration | No | The date of admission (2024-09-05) falls outside the policy coverage period (2024-01-01 to 2024-12-31). |\n| Doctor specialization | Yes | Pulmonologist is a valid specialization for diagnosing and treating lung cancer. |\n| The diagnosis is consistent | Yes | The diagnosis of lung cancer is supported by the CT scan and biopsy results. |\n| Treatment necessary | No | While the treatment plan is typical for lung cancer, additional information is needed to confirm the necessity of the specific treatment proposed in this case. The claim requires further review to determine if the proposed treatment is medically necessary. |\n\n**Claim Rejected** \n\n**Reason for Rejection:** The date of admission falls outside the policy coverage period. Additionally, while the proposed treatment plan is common for lung cancer, additional information is required to confirm the medical necessity of the specific treatment in this case.  The claim requires further review to determine if the treatment is medically necessary. \n",
      ],
    },
  ]
)

def query_icd_code(query, top_k=1):
    query_embedding = embedding_model.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=top_k)
    return results  

def extract_details_from_pdf(pdf_path):
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

def check_document_consistency(summary_file_path):
    extracted_sections = extract_details_from_pdf(summary_file_path)
    diagnosis_details = extracted_sections.get("Diagnosis Details", "")
    treatment_summary = extracted_sections.get("Treatment Summary", "")
    test_text = diagnosis_details + treatment_summary

    inputs = tokenizer_cons(test_text, return_tensors="pt").to(device)
    with torch.no_grad():  
        outputs = model_cons(**inputs)  
    prediction = outputs.logits.argmax(-1).item()

    return prediction == 0

def check_document_neccessary(summary_file_path):
    extracted_sections = extract_details_from_pdf(summary_file_path)
    diagnosis_details = extracted_sections.get("Diagnosis Details", "")
    treatment_summary = extracted_sections.get("Treatment Summary", "")
    test_text = diagnosis_details + treatment_summary

    inputs = tokenizer_nec(test_text, return_tensors="pt").to(device)
    with torch.no_grad():  
        outputs = model_nec(**inputs)  
    prediction = outputs.logits.argmax(-1).item()

    return prediction == 0  

def check_specialization(diagnosis, specialization):
    diagnosis_lower = diagnosis.lower()
  
    for condition, specializations in valid_specializations.items():
        if condition in diagnosis_lower:
            if specialization in specializations:
                return True
            else:
                return False
  
    return False

def extract_icd_codes_with_ner_and_regex(text):
    icd_codes = []
    icd_pattern = r'\b[A-Z]\d{2}\.\d\b|\b[A-Z]\d{2}\b'
    icd_codes = re.findall(icd_pattern, text)

    return icd_codes

def extract_specialization(text):
    pattern = r"Specialization:\s*([A-Za-z]+)"
    match = re.search(pattern, text)
    
    if match:
        specialization = match.group(1)
    else:
        specialization = "Pulmonologist"  # Handle if not found
    
    return specialization
    
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def parse_pdf_text(pdf_text):
    lines = pdf_text.splitlines()
    policy_number = None
    total_hospital_bill = 0
    total_pharmacy_bill = 0

    for line in lines:
        if "Insurance Policy Number:" in line:
            policy_number = line.split("Insurance Policy Number:")[-1].strip()
        elif "Total Hospital Bill:" in line:
            total_hospital_bill = int(line.split("₹")[-1].replace(",", "").strip())
        elif "Total Pharmacy Bill:" in line:
            total_pharmacy_bill = int(line.split("₹")[-1].replace(",", "").strip())

    return policy_number, total_hospital_bill, total_pharmacy_bill


def process_claim(policy_number, total_hospital_bill, total_pharmacy_bill, max_claimable):
        
    grand_total = total_hospital_bill + total_pharmacy_bill
        
    if grand_total == max_claimable:
        covered_amount = max_claimable  # All the bill is covered
        patient_pays = 0  # Patient doesn't need to pay anything
        remaining_claimable = 0  
        return grand_total, covered_amount, patient_pays, remaining_claimable

    elif grand_total < max_claimable:
        covered_amount = 0.8 * grand_total
        patient_pays = grand_total - covered_amount
        remaining_claimable = max_claimable - covered_amount  
        return grand_total, covered_amount, patient_pays, remaining_claimable

    else:
        covered_amount = max_claimable  # Coverage is capped at max claimable amount
        patient_pays = grand_total - max_claimable  # Patient pays the remaining amount
        remaining_claimable = 0  # All the claimable amount is used up
        return grand_total, covered_amount, patient_pays, remaining_claimable
    
def pdf_whole_text(path):
    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()
    
    return extracted_text

def generate_reason_report(text):
    # Replace rupee symbol with "Rs."
    text = text.replace('\u20b9', 'Rs.')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    bold_sections = ['Claim Summary Report', 'Patient Information', 'Doctor Information', 'Hospital Details', 'Diagnosis Summary', 'Treatment Plan', 'Eligibility Check', 'Claim Pending', 'Claim Rejected', 'Claim Partially Approved', 'Bill Summary', 'Claim Approved']
    text = re.sub(r'[\*#]', '', text)
    lines = text.split("\n")
    pdf.set_font('Arial', '', 12)

    for line in lines:
        line = line.strip()
        if any(bold_section in line for bold_section in bold_sections):
            pdf.set_font('Arial', 'B', 12)  
            pdf.cell(0, 10, line, ln=True)
        elif "|" in line:  
            pdf.set_font('Arial', '', 12)
            cols = line.split("|")[1:-1] 

            col_widths = [60, 40, 90]
            for i, col in enumerate(cols):
                pdf.cell(col_widths[i], 10, col.strip(), border=1, align='C' if i == 1 else 'L')
            pdf.ln()
        else:
            pdf.set_font('Arial', '', 12) 
            pdf.multi_cell(0, 6, line) 
        pdf.ln(4)

    # Set the fixed file name as "claim_summary.pdf"
    reason_report = os.path.join("static/img", "claim_summary.pdf")
    pdf.output(reason_report)

    return reason_report

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/raise_claim')
def raise_claim():
    return render_template('form.html')

@app.route('/submit_claim', methods=['GET', 'POST'])
def submit_claim():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile_number = request.form.get('mobile_number')
        insurance_policy_number = request.form.get('insurance_policy_number')
        dob = request.form.get('dob')
        admission_type = request.form.get('type')
        hospital_name = request.form.get('hospital_name')
        hospital_id = request.form.get('hospital_id')
        date_of_admission = request.form.get('date_of_admission')
        date_of_discharge = request.form.get('date_of_discharge')
        pre_authorization = request.form.get('pre_authorization')

        summary_file = request.files.get('summary_document')
        id_proof_file = request.files.get('id_proof')

        if not summary_file or not id_proof_file:
            flash('Both files are required.', 'error')
            return redirect(url_for('submit_claim'))

        summary_file_path = os.path.join(app.config['UPLOAD_FOLDER'], summary_file.filename)
        id_proof_path = os.path.join(app.config['UPLOAD_FOLDER'], id_proof_file.filename)
        summary_file.save(summary_file_path)
        id_proof_file.save(id_proof_path)

        has_policy = bool(policies_collection.find_one({'policy_number': insurance_policy_number}))
        
        if not has_policy:
            claim_status = {
                "has_policy": has_policy,
                "dates_covered": False,
                "diagnosis_covered": False,
                "doctor_checked": False,
                "consistency_checked": False,
                "necessary_checked": False
            }
            
            has_policy_message = "The patient has a policy" if claim_status["has_policy"] else "The patient has no policy"
            response = chat_session.send_message(pdf_whole_text(summary_file_path) + has_policy_message)
            gen_out_text = response.text
            reason_report = generate_reason_report(gen_out_text)
            
            return render_template('animate.html', claim_status=claim_status)
        
        date_of_admission = datetime.strptime(date_of_admission, '%Y-%m-%d')
        date_of_discharge = datetime.strptime(date_of_discharge, '%Y-%m-%d')
        policy_date = policies_collection.find_one({'policy_number': insurance_policy_number})
        effective_date = policy_date.get('effective_date')
        expiration_date = policy_date.get('expiration_date')
        effective_date = datetime.strptime(effective_date, '%Y-%m-%d') 
        expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    
        dates_covered = effective_date <= date_of_admission <= expiration_date and effective_date <= date_of_discharge <= expiration_date

        if not dates_covered:
            claim_status = {
                "has_policy": True,
                "dates_covered": False,
                "diagnosis_covered": False,
                "doctor_checked": False,
                "consistency_checked": False,
                "necessary_checked": False
                
            }
            date_message = "Admission and discharge dates are not covered by the policy."
            response = chat_session.send_message(pdf_whole_text(summary_file_path) + date_message)
            gen_out_text = response.text
            reason_report = generate_reason_report(gen_out_text)
            return render_template('animate.html', claim_status=claim_status)

        else:
            policy_doc = policies_collection.find_one({'policy_number': insurance_policy_number})
            document_path = policy_doc.get('policy_document')
            policy_text = extract_text_from_docx(document_path)
            policy_icd_codes = extract_icd_codes_with_ner_and_regex(policy_text)

            extracted_sections = extract_details_from_pdf(summary_file_path)
            diagnosis_details = extracted_sections.get("Diagnosis Details", "")
            icd_code = query_icd_code(diagnosis_details)
            icd = icd_code['matches'][0]['id'] if icd_code['matches'] else 0

            full_txt = pdf_whole_text(summary_file_path)
            doc_spec = extract_specialization(full_txt)

            diagnosis_covered = icd in policy_icd_codes
            doctor_checked = check_specialization(diagnosis_details, doc_spec)
            print(doc_spec)
            consistency_checked = check_document_consistency(summary_file_path)
            necessary_checked = check_document_neccessary(summary_file_path)
            
            claim_status = {
                "has_policy": has_policy,
                "dates_covered": dates_covered,
                "diagnosis_covered": diagnosis_covered,
                "doctor_checked": doctor_checked,
                "consistency_checked": consistency_checked,
                "necessary_checked": necessary_checked
            }
            
            has_policy_message = "The patient has a policy. " if claim_status["has_policy"] else "The patient has no policy. "
            dates_covered_message = "The dates mentioned is under the expiration. " if claim_status["dates_covered"] else "The dates are not covered under the expiration date. "
            diagnosis_covered_message = "The diagnosis is covered under policy. " if claim_status["diagnosis_covered"] else "The diagnosis is not covered under policy. "
            doctor_checked = "The doctor specialization is valid for the diagnosis. " if claim_status["doctor_checked"] else "The doctor specialization is invalid for the diagnosis. "
            consistency_checked_message = "The diagnosis is consistent. " if claim_status["consistency_checked"] else "The diagnosis is inconsistent. "
            necessary_checked_message = "The treatment is necessary. " if claim_status["necessary_checked"] else "The treatment is experimental. "

            cov_limit = policy_doc.get('coverage_limits')
            pdf_text = extract_text_from_pdf(summary_file_path)
            pol_number, total_hospital_bill, total_pharmacy_bill = parse_pdf_text(pdf_text)
            grand_total, covered_amount, patient_pays, remaining_claimable = process_claim(pol_number, total_hospital_bill, total_pharmacy_bill, cov_limit)
            
            bills_cost = " Grand total: "+ "Rs." + str(grand_total) + " Covered Amount: Rs." + str(covered_amount) + " patient payable: Rs." + str(patient_pays) + " remaining claimable: Rs." + str(remaining_claimable)
            
            gen_inp_text = pdf_whole_text(summary_file_path)
            gen_inp_text = gen_inp_text + has_policy_message  + dates_covered_message + diagnosis_covered_message + doctor_checked + consistency_checked_message + necessary_checked_message + bills_cost
            response = chat_session.send_message(gen_inp_text)
            gen_out_text = response.text
            generate_reason_report(gen_out_text)
            reason_report = generate_reason_report(gen_out_text)

            patient_data = {
                'patient_name': patient_name,
                'age': age,
                'dob': dob,
                'gender': gender,
                'mobile_number': mobile_number,
                'insurance_policy_number': insurance_policy_number,
                'type': admission_type,
                'hospital_name': hospital_name,
                'hospital_id': hospital_id,
                'date_of_admission': date_of_admission,
                'date_of_discharge': date_of_discharge,
                'pre_authorization': pre_authorization,
                'summary_document': summary_file_path,
                'id_proof': id_proof_path
            }

            patients_collection.insert_one(patient_data)

            return render_template('animate.html', claim_status=claim_status)

    return render_template('form.html')


@app.route('/animate')
def animate():
    return render_template('animate.html')

@app.route('/error')
def error():
    reason_report = request.args.get('reason_report')
    return render_template('error.html', pdf_file_name = reason_report)

@app.route('/success')
def success():
    reason_report = request.args.get('reason_report')
    return render_template('success.html', pdf_file_name= reason_report)

if __name__ == '__main__':
    app.run(debug=True)