import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def main():
    def read_pdf_paragraphs_with_pdfplumber(pdf_path) -> list[str]:
        paragraphs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                page_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                paragraphs.extend(page_paragraphs)

        for i, paragraph in enumerate(paragraphs):
            print(f"--- Paragraph {i + 1} ---")
            print(paragraph)

        return paragraphs

    pdf_path = "konstytucja.pdf"
    para_read = read_pdf_paragraphs_with_pdfplumber(pdf_path)
    if not para_read:
        raise ValueError("No text extracted from the PDF.")

    documents = [Document(page_content=str(text)) for text in para_read]

    text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    #print(para_read)
    split_documents = text_splitter.split_documents(documents)
    #print(f"Split documents: {len(split_documents)}")
    #print(split_documents)
    """split_texts = []
    for text in documents:
        split_texts.extend(text_splitter.split_documents(text))"""


    """for doc in split_documents[:2]:  # Print first 2 documents
        print(f"Chunk size: {len(doc.page_content)}")
        print(doc.page_content[:100] + "...")"""
    #documents = [Document(page_content=str(text)) for text in split_documents]

    return split_documents
if __name__ == "__main__":
    main()


