import requests
from pypdf import PdfReader

class Page:
    def __init__(self, text: str, images: list[bytes]):
        self.texts = text
        self.images = images

def extract_text_and_images_from_paper(file_path: str) -> list[Page]:
    '''
    Extracts text and images from PDF file pages
    
    Args:
        file_path: Path to a PDF file (research paper)
    Return:
        List of pages (where a page object contains page text and bytes of the images)
    '''
    reader = PdfReader(file_path)
    output = []
    for page in reader.pages:
        text = page.extract_text()
        images = [img.data for img in page.images]
        output.append(Page(text, images))
    return output

if __name__ == '__main__':
    url = "https://arxiv.org/pdf/1706.03762"
    pdf_file = "paper.pdf"
    response = requests.get(url)
    response.raise_for_status() 
    with open(pdf_file, "wb") as f:
        f.write(response.content)
    pages = extract_text_and_images_from_paper(pdf_file)
    print(pages[2].texts[50:])
    print(pages[2].images[0][10:])
