"""
text_detection_module
"""

from typing import List
from PIL import Image

def detect_text(image: Image, languages: List[str] = None) -> str:
    """
    Detects and extracts text from a given image.

    Parameters:
        image (PIL.Image): The input image to extract text from.
        languages (List[str], optional): A list of language codes to detect text in.
            Defaults to ["en"]. Supported languages can be found at:
            https://github.com/VikParuchuri/surya/blob/master/surya/languages.py

    Returns:
        str: The extracted text from the image.

    Notes:
        Model weights will automatically download the first time you run this function.
    """
    if languages is None:
        languages = ["en"]

    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError:
        raise ImportError(
            "The required dependencies for text detection are not installed. "
            "Please install them using `pip install scrapegraphai[screenshot_scraper]`."
        )

    # Load models and processors for detection and recognition
    det_processor = load_det_processor()
    det_model = load_det_model()
    rec_processor = load_rec_processor()
    rec_model = load_rec_model()

    # Run OCR and collect predictions
    predictions = run_ocr([image], [languages], det_model, det_processor, rec_model, rec_processor)
    
    # Extract text from predictions
    extracted_text = "\n".join([line.text for line in predictions[0].text_lines])
    return extracted_text
