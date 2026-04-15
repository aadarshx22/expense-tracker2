import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import Optional

    #Image Processing for OCR maximuma accuracy 
def preprocess_image(image_bytes: bytes) -> np.ndarray:

    # Decode bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Resize if too small (Tesseract works best at 300+ DPI equivalent)
    h, w = gray.shape
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. Deskew (fix tilted receipts)
    gray = deskew(gray)

    # 4. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 5. Adaptive threshold — works better than global threshold for uneven lighting
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 6. Morphological cleanup — remove tiny noise specks
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    return processed


def deskew(image: np.ndarray) -> np.ndarray:
    """Detect and correct rotation using Hough lines."""
    try:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return image

        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = (theta - np.pi / 2) * 180 / np.pi
            if abs(angle) < 45:  # Only small rotations
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        if abs(median_angle) < 0.5:  # Skip tiny corrections
            return image

        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image  # If deskew fails, return original


def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Full OCR pipeline: preprocess → extract → parse structure.
    Returns raw text + confidence + basic parsed fields.
    """
    # Preprocess
    processed = preprocess_image(image_bytes)

    # Convert to PIL for pytesseract
    pil_image = Image.fromarray(processed)

    # OCR config:
    # --oem 3 = LSTM + legacy engine (best accuracy)
    # --psm 6 = Assume uniform block of text (good for receipts)
    custom_config = r'--oem 3 --psm 6'

    # Get text with confidence data
    ocr_data = pytesseract.image_to_data(
        pil_image,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    # Calculate average confidence (filter out -1 values which are layout markers)
    confidences = [int(c) for c in ocr_data['conf'] if int(c) != -1]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Get clean full text
    raw_text = pytesseract.image_to_string(pil_image, config=custom_config)
    raw_text = raw_text.strip()

    # Basic structural extraction with regex
    parsed = parse_receipt_structure(raw_text)

    return {
        "raw_text": raw_text,
        "confidence": round(avg_confidence, 2),
        "parsed": parsed,
        "word_count": len(raw_text.split())
    }


def parse_receipt_structure(text: str) -> dict:
    """
    Regex-based extraction of common receipt fields.
    This is a fast pre-pass; Claude will do the deep categorization.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Amount patterns: $12.34 or 12.34 or Rs. 234
    amount_pattern = re.compile(
        r'(?:total|amount|grand total|subtotal|sum)[:\s]*'
        r'(?:rs\.?|inr|₹|\$|€|£)?\s*(\d{1,6}[.,]\d{2})',
        re.IGNORECASE
    )

    # Date patterns
    date_pattern = re.compile(
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b'
    )

    # Currency amount (standalone)
    currency_pattern = re.compile(
        r'(?:rs\.?|inr|₹|\$|€|£)\s*(\d{1,6}[.,]\d{2})|(\d{1,6}[.,]\d{2})\s*(?:rs\.?|inr|₹|\$|€|£)',
        re.IGNORECASE
    )

    # Extract fields
    amounts = amount_pattern.findall(text.lower())
    dates = date_pattern.findall(text)
    all_currency = currency_pattern.findall(text)

    # Flatten and clean currency matches
    currency_values = []
    for match in all_currency:
        val = match[0] or match[1]
        if val:
            currency_values.append(float(val.replace(',', '.')))

    # Try to identify merchant (usually first non-empty line)
    merchant = lines[0] if lines else None

    # Total = largest amount found (heuristic)
    total = max(currency_values) if currency_values else None

    return {
        "merchant": merchant,
        "date": dates[0] if dates else None,
        "total": total,
        "all_amounts": sorted(set(currency_values), reverse=True)[:10],
        "line_items_raw": lines  # Raw lines for Claude to parse
    }