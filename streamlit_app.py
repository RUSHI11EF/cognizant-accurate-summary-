import streamlit as st
import PyPDF2
import pickle
import ast
from langchain_groq import ChatGroq
from rouge_score import rouge_scorer
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API keys from the environment
groq_api_key = os.getenv('GROQ_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

# Load the model configuration
with open('model_config.pkl', 'rb') as config_file:
    model_config = pickle.load(config_file)

# Initialize the model with the API keys
model = ChatGroq(
    model=model_config['model_name'], 
    max_tokens=model_config['max_tokens'], 
    groq_api_key=groq_api_key,  # Loaded from the .env file
    langchain_api_key=langchain_api_key  # Loaded from the .env file
)

# Streamlit app
st.title("Medical Report Summarization")

# File uploader for the PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    report = ""
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        report += text
    
    # Display the extracted text (optional)
    st.text_area("Extracted Text from PDF", value=report, height=200)

    if st.button("Generate Summary"):
        # Prepare the prompt
        messages = [
            {
                "role": "system",
                "content": model_config['summary_prompt'],
            },
            {
                "role": "user",
                "content": report,
            },
        ]

        # Generate the summary
        summary = model.invoke(messages)
        summary_d = summary.content.strip("```")
        summary_dict = ast.literal_eval(summary_d)

        # Display the summary
        st.json(summary_dict)

        # Display the ROUGE scores (if reference summary is provided)
        reference_summary_str = st.text_area("Enter the reference summary (optional):", height=200)
        if reference_summary_str:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            generated_summary_str = str(summary_dict)
            scores = scorer.score(reference_summary_str, generated_summary_str)

            st.write("ROUGE-1:", scores['rouge1'])
            st.write("ROUGE-2:", scores['rouge2'])
            st.write("ROUGE-L:", scores['rougeL'])
