import PyPDF2
import re
import random

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages])
    return text

def extract_abstracts(text):
    abstracts = []
    matches = re.findall(r'(Objective.*?)\n\s*Conflict of Interest', text, re.DOTALL)
    
    for abstract in matches:
        abstracts.append(abstract.strip())
    
    return abstracts

if __name__ == "__main__":
    pdf_path = "PRC2022Abstracts.pdf"
    text = extract_text_from_pdf(pdf_path)
    abstracts_data = extract_abstracts(text)

    sample_abstracts = random.sample(abstracts_data, 20)