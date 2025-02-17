# parser.py

import os
import re
import pdfplumber
import html2text
import docx
from bs4 import BeautifulSoup

def parse_document(file_path):
    """
    Parses the given document file (HTML, PDF, DOCX, or TXT) and extracts clean text.
    
    :param file_path: str - Path to the document file.
    
    :return: str - The extracted and cleaned text content.
    """
    
    # Ensure the file exists before processing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = file_path.lower().split('.')[-1]

    if file_extension in ["html", "htm"]:
        return parse_html(file_path)
    elif file_extension == "pdf":
        return parse_pdf(file_path)
    elif file_extension in ["docx"]:
        return parse_docx(file_path)
    elif file_extension in ["txt", "md"]:
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def parse_html(file_path):
    """
    Parses an HTML file and extracts readable text.
    
    :param file_path: str - Path to the HTML file.
    
    :return: str - Extracted clean text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    # Convert HTML to markdown-style text
    text = html2text.html2text(soup.prettify())
    
    # Remove excessive whitespace and return clean text
    return clean_text(text)

def parse_pdf(file_path):
    """
    Parses a PDF file and extracts readable text.
    
    :param file_path: str - Path to the PDF file.
    
    :return: str - Extracted clean text.
    """
    extracted_text = []
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    
    text = "\n".join(extracted_text)
    return clean_text(text)

def parse_docx(file_path):
    """
    Parses a DOCX file and extracts readable text.
    
    :param file_path: str - Path to the DOCX file.
    
    :return: str - Extracted clean text.
    """
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    
    return clean_text(text)

def parse_text(file_path):
    """
    Parses a plain text file and extracts readable text.
    
    :param file_path: str - Path to the TXT file.
    
    :return: str - Extracted clean text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return clean_text(text)

def clean_text(text):
    """
    Cleans extracted text by removing extra whitespace, unwanted symbols, and excessive newlines.
    
    :param text: str - Raw extracted text.
    
    :return: str - Cleaned text.
    """
    text = text.strip()
    
    # Remove multiple newlines and excessive spaces
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Collapse multiple newlines
    text = re.sub(r"[ ]{2,}", " ", text)     # Reduce multiple spaces to one

    return text
