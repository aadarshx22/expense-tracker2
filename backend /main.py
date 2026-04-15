from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn
import pytesseract
import cv2
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Receipt OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
MAX_SIZE_MB = 10




def preprocess_image_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]

    return thresh




def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9.\n ]', '', text)
    return text


def extract_amount(text):
    lines = text.split('\n')

    for line in lines:
        if "total" in line:
            nums = re.findall(r'\d+\.\d{2}', line)
            if nums:
                return float(nums[-1])

    amounts = re.findall(r'\d+\.\d{2}', text)
    return max(map(float, amounts)) if amounts else None


def extract_date(text):
    match = re.search(r'\d{2}/\d{2}/\d{4}', text)
    return match.group() if match else None


def extract_vendor(text):
    lines = text.split('\n')

    for line in lines[:5]:
        if len(line.strip()) > 3:
            return line.strip()

    return "Unknown"






def extract_text_from_image(image_bytes):
    image = preprocess_image_from_bytes(image_bytes)

    config = r'--oem 3 --psm 6'

    raw_text = pytesseract.image_to_string(image, config=config)

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    confidences = [int(conf) for conf in data["conf"] if conf != '-1']
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    cleaned = clean_text(raw_text)

    parsed = {
    "vendor": extract_vendor(cleaned),
    "amount": extract_amount(cleaned),
    "date": extract_date(cleaned),
    }


    category_data = categorize_receipt(raw_text)

    parsed["category"] = category_data["category"]
    parsed["subcategory"] = category_data["subcategory"]
    parsed["mcc"] = category_data["mcc"]
    parsed["category_confidence"] = category_data["confidence"]
    parsed["category_method"] = category_data["method"]
    parsed["merchant_detected"] = category_data["merchant"]

    return {
        "raw_text": raw_text,
        "confidence": round(avg_conf, 2),
        "word_count": len(raw_text.split()),
        "parsed": parsed
    }


@app.get("/health")
async def health():
    try:
        version = pytesseract.get_tesseract_version()
        return {"status": "ok", "tesseract_version": str(version)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/ocr/extract")
async def extract(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_SIZE_MB}MB.")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")

    try:
        result = extract_text_from_image(image_bytes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(500, f"OCR processing failed: {str(e)}")

    return JSONResponse({
        "filename": file.filename,
        "ocr_confidence": result["confidence"],
        "word_count": result["word_count"],
        "raw_text": result["raw_text"],
        "parsed": result["parsed"]
    })



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
