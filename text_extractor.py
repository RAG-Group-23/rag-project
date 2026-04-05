from pypdf import PdfReader
import requests

def extract_text_from_paper(file_path: str) -> list[str]:
    '''
    Extracts text from PDF file pages
    
    Args:
        file_path: Path to a PDF file (research paper)
    Return:
        List of strings per page in PDF
    '''
    reader = PdfReader(file_path)
    page_texts = []
    for page in reader.pages:
        text = page.extract_text()
        page_texts.append(text)
    return page_texts


if __name__ == '__main__':
    url = "https://arxiv.org/pdf/1706.03762"
    pdf_file = "paper.pdf"
    response = requests.get(url)
    response.raise_for_status() 
    with open(pdf_file, "wb") as f:
        f.write(response.content)
    pages = extract_text_from_paper(pdf_file)
    print(f"Extracted text for {len(pages)} pages")
    for i, page in enumerate(pages[:3]): 
        print(f"--- Page {i+1} ---")
        print(page[:1000])  
        print("\n")