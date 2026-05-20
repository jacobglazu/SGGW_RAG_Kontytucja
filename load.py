import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf_paragraphs_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            paragraphs = text.split("\n\n")  # Split by paragraphs
            for i, paragraph in enumerate(paragraphs):
                print(f"--- Paragraph {i + 1} ---")
                print(paragraph)
                # Process the paragraph here
                return paragraphs

# Example usage
pdf_path = "kostytucja.pdf"
para_read = read_pdf_paragraphs_with_pdfplumber(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
split_documents = text_splitter.split_documents(para_read)
