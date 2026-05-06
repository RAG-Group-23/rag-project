"""
text_extractor.py — Extract per-page text and images from PDF files.
"""

from pypdf import PdfReader


class Page:
    """Holds the extracted text and raw image bytes for a single PDF page."""

    def __init__(self, text: str, images: list[bytes]):
        self.texts = text
        self.images = images


def extract_text_and_images_from_paper(file_path) -> list[Page]:
    """
    Extract text and images from each page of a PDF.

    Parameters
    ----------
    file_path : str | pathlib.Path | file-like object
        Path to a PDF file, or any file-like object accepted by PdfReader
        (e.g. ``io.BytesIO``).

    Returns
    -------
    list[Page]
        One ``Page`` per PDF page, each containing the page's text and a list
        of raw image bytes.
    """
    reader = PdfReader(file_path)
    output = []
    for page in reader.pages:
        text = page.extract_text() or ""
        images = [img.data for img in page.images]
        output.append(Page(text, images))
    return output
