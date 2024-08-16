# -*- coding: utf-8 -*-
"""lama_Summarizer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1j-3lDdHQDc2iNy9QJwFLvliWZs_pVab7
"""

!pip3 install PyPDF2
!pip install langchain
!pip install -qU langchain-groq
!pip install langchain_community
!pip install rouge-score
!pip install transformers

import PyPDF2
import os
import ast
from google.colab import userdata
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

!pip install PyCryptodome

import PyPDF2


report = ""

# Open the PDF file
with open('/content/Sample-filled-in-MR.pdf', 'rb') as file:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(file)

    # Get the number of pages in the PDF
    num_pages = len(pdf_reader.pages)
    print(f"The PDF has {num_pages} pages.")

    # Extract the text content from each page
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        report += text

print(report)

import os
from google.colab import userdata
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = userdata.get('LANGCHAIN_API_KEY')
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

!pip install langchain_groq

from langchain_groq import ChatGroq

model = ChatGroq(model="llama-3.1-70b-versatile", max_tokens=8000)

import ast
from rouge_score import rouge_scorer

from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

messages = [
    SystemMessage(content="""You have to summarize the Medical report.
    You should be able to specify the important details specifically like patient age, the complaint and the diseases suffered by the patient.
    The doctor who reads the summary should able to remember the diagnosis done to the patient previously.

    Example JSON:

    {{
        "name" : Name of the patient,
        "age" : Age of the patient
        "prescription" : Medicines prescribed by the doctor,
        "Gender": Gender of the patient.
        "Medical History": Brief overview of the patient's relevant medical history.
        "Allergies": Any known allergies the patient has.
        "Symptoms": Key symptoms presented by the patient.
        "Vital Signs": Basic vital signs like blood pressure, heart rate, temperature, etc.
        "Tests Conducted" : List of tests conducted, if any, and their results.
        "Treatment Plan" : Outline of the treatment plan.
        "Follow-Up" : Scheduled follow-up date or required next steps.
        "Contact Information" : Patient's contact details for easy reference.
        "Doctor's Notes" : Additional observations or important notes by the doctor.
         "Date of Report" : Date when the report was created or updated.
        "short_summary" : Short summary of the report,
        "diagnosis" : Diagnosis done to the patient
    }}

    Do not start with Here is the summary....

    You have to output a STRICTLTY JSON ONLY.
    Do not output anything else. NO MARKDOWN JUST A STRING
    """),
    HumanMessage(content=f"{report}"),
]

summary = model.invoke(messages)



summary_d = summary.content

import ast
summary_d = summary_d.strip("```")
summary_dict = ast.literal_eval(summary_d)
summary_dict

# Define the reference summary
reference_summary = {
    "name": "Alice Smith",
    "age": 52,
    "prescription": "Some medication",
    "Gender": "Female",
    "Medical History": "Heart condition",
    "Allergies": "None mentioned",
    "Symptoms": "Cough",
    "Vital Signs": "Stable",
    "Tests Conducted": "General tests",
    "Treatment Plan": "Basic care",
    "Follow-Up": "Check-up in a week",
    "Contact Information": "Provided",
    "Doctor's Notes": "No significant concerns",
    "Date of Report": "2023-08-11",
    "short_summary": "Patient with a heart condition and a cough. Given basic medication.",
    "diagnosis": "Common cold"
}

# Convert both summaries to strings for ROUGE evaluation
generated_summary = str(summary_dict)
reference_summary_str = str(reference_summary)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Calculate ROUGE scores
scores = scorer.score(reference_summary_str, generated_summary)

# Display ROUGE scores
print("ROUGE-1:", scores['rouge1'])
print("ROUGE-2:", scores['rouge2'])
print("ROUGE-L:", scores['rougeL'])

import pickle

# Save the model and its configuration
model_config = {
    'model_name': "llama-3.1-70b-versatile",
    'max_tokens': 8000,
    'summary_prompt': """You have to summarize the Medical report.
        You should be able to specify the important details specifically like patient age, the complaint and the diseases suffered by the patient.
        The doctor who reads the summary should able to remember the diagnosis done to the patient previously.

        Example JSON:

        {{
            "name" : Name of the patient,
            "age" : Age of the patient,
            "prescription" : Medicines prescribed by the doctor,
            "Gender": Gender of the patient,
            "Medical History": Brief overview of the patient's relevant medical history,
            "Allergies": Any known allergies the patient has,
            "Symptoms": Key symptoms presented by the patient,
            "Vital Signs": Basic vital signs like blood pressure, heart rate, temperature, etc.,
            "Tests Conducted" : List of tests conducted, if any, and their results,
            "Treatment Plan" : Outline of the treatment plan,
            "Follow-Up" : Scheduled follow-up date or required next steps,
            "Contact Information" : Patient's contact details for easy reference,
            "Doctor's Notes" : Additional observations or important notes by the doctor,
            "Date of Report" : Date when the report was created or updated,
            "short_summary" : Short summary of the report,
            "diagnosis" : Diagnosis done to the patient
        }}

        Do not start with Here is the summary....

        You have to output a STRICTLY JSON ONLY.
        Do not output anything else. NO MARKDOWN JUST A STRING
    """
}

# Save the configuration
with open('model_config.pkl', 'wb') as config_file:
    pickle.dump(model_config, config_file)

print("Model configuration saved successfully.")

model.save_pretrained('/content/summarization_model')

