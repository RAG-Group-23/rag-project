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