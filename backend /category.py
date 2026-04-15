import re
from collections import defaultdict

MERCHANT_MCC_MAP = {
    # Travel
    "uber": 4121,
    "ola": 4121,
    "irctc": 4112,
    "indigo": 4511,
    "air india": 4511,
    "spicejet": 4511,
    "redbus": 4131,
    "oyo": 7011,

    # Food
    "zomato": 5812,
    "swiggy": 5814,
    "dominos": 5814,
    "pizza hut": 5814,
    "mcdonalds": 5814,
    "starbucks": 5812,
    "ccd": 5812,
    "haldiram": 5812,

    # Shopping
    "amazon": 5311,
    "flipkart": 5311,
    "myntra": 5651,
    "ajio": 5651,
    "dmart": 5411,
    "reliance fresh": 5411,
    "croma": 5732,
    "vijay sales": 5732,
    "tanishq": 5944,

    # Utilities
    "airtel": 4814,
    "jio": 4814,
    "bsnl": 4814,
    "tata power": 4900,
    "adani electricity": 4900,

    # Healthcare
    "apollo pharmacy": 8062,
    "fortis": 8062,
    "max hospital": 8062,

    # Fuel
    "bharat petroleum": 5541,
    "indian oil": 5541,
    "hp petrol": 5541,

    # Government / Others
    "india post": 4215,
    "municipal": 9399,
}

MCC_KEYWORDS = {
    5812: ["restaurant", "dine", "cafe", "food", "thali"],
    5814: ["pizza", "burger", "fast food", "snack"],
    4112: ["train", "railway", "irctc"],
    4121: ["taxi", "cab", "auto", "uber", "ola"],
    4511: ["flight", "airlines", "boarding", "airport"],
    4131: ["bus", "ticket"],
    7011: ["hotel", "lodging", "resort", "stay"],
    5411: ["grocery", "supermarket", "provision"],
    5732: ["electronics", "mobile", "laptop", "tv"],
    5541: ["petrol", "diesel", "fuel"],
    4900: ["electricity", "water", "gas", "utility", "bill"],
    4814: ["recharge", "mobile", "broadband", "telecom", "wifi"],
    8062: ["hospital", "clinic", "pharmacy", "medical", "doctor"],
    4215: ["courier", "delivery", "logistics"],
    6513: ["rent", "lease", "property"],
    9399: ["tax", "government", "municipal", "challan"],
}
MCC_CATEGORY_MAP = {
    # Travel & Transportation
    4511: ("Travel and Transportation", "Airlines & Air Carriers"),
    4112: ("Travel and Transportation", "Passenger Railways"),
    4121: ("Travel and Transportation", "Taxi-cabs & Limousines"),
    4131: ("Travel and Transportation", "Bus Lines"),
    4784: ("Travel and Transportation", "Tolls & Bridge Fees"),
    7011: ("Travel and Transportation", "Lodging"),

    # Food and Dining
    5812: ("Food and Dining", "Eating Places & Restaurants"),
    5814: ("Food and Dining", "Fast Food Restaurants"),
    5813: ("Food and Dining", "Drinking Places"),
    5462: ("Food and Dining", "Bakeries"),

    # Shopping and Retail
    5411: ("Shopping and Retail", "Grocery Stores/Supermarkets"),
    5311: ("Shopping and Retail", "Department Stores"),
    5651: ("Shopping and Retail", "Clothing Stores"),
    5691: ("Shopping and Retail", "Clothing Stores"),
    5944: ("Shopping and Retail", "Jewelry, Watches, & Silverware"),
    5732: ("Shopping and Retail", "Electronics Shops"),
    5722: ("Shopping and Retail", "Electronics Shops"),

    # Utilities and Services
    4900: ("Utilities and Services", "Electricity/Gas/Water/Sanitary"),
    4814: ("Utilities and Services", "Telecommunication Services"),
    4899: ("Utilities and Services", "Cable and Pay TV"),
    8062: ("Utilities and Services", "Healthcare/Hospitals"),
    8299: ("Utilities and Services", "Schools & Educational Services"),

    # Specialized Vendor Payments
    5541: ("Specialized Vendor Payments", "Fuel/Service Stations"),
    9399: ("Specialized Vendor Payments", "Government Services"),
    4215: ("Specialized Vendor Payments", "Courier Services"),
    6513: ("Specialized Vendor Payments", "Real Estate Agents"),
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_merchant(ocr_text: str) -> str:
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    noise_words = {"invoice", "receipt", "bill", "tax", "gst"}

    for line in lines:
        cleaned = normalize_text(line)
        if cleaned and not any(word in cleaned for word in noise_words):
            return cleaned
    return "unknown"

def get_mcc_from_merchant(merchant: str):
    merchant_norm = normalize_text(merchant)
    for name, mcc in MERCHANT_MCC_MAP.items():
        if name in merchant_norm:
            return mcc
    return None

def get_mcc_from_keywords(ocr_text: str):
    text = normalize_text(ocr_text)
    scores = defaultdict(int)

    for mcc, keywords in MCC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[mcc] += 1

    if not scores:
        return None, 0.0

    best_mcc = max(scores, key=scores.get)
    confidence = scores[best_mcc] / sum(scores.values())
    return best_mcc, round(confidence, 2)

def categorize_receipt(ocr_text: str):
    merchant = extract_merchant(ocr_text)

    # Step 1: Merchant-based MCC
    mcc = get_mcc_from_merchant(merchant)
    if mcc:
        category, subcategory = MCC_CATEGORY_MAP[mcc]
        return {
            "merchant": merchant.title(),
            "mcc": mcc,
            "category": category,
            "subcategory": subcategory,
            "confidence": 1.0,
            "method": "merchant_mcc"
        }

    # Step 2: Keyword-based MCC (Fallback)
    mcc, confidence = get_mcc_from_keywords(ocr_text)
    if mcc:
        category, subcategory = MCC_CATEGORY_MAP[mcc]
        return {
            "merchant": merchant.title(),
            "mcc": mcc,
            "category": category,
            "subcategory": subcategory,
            "confidence": confidence,
            "method": "keyword_mcc"
        }

    # Step 3: Final fallback
    return {
        "merchant": merchant.title(),
        "mcc": None,
        "category": "Others",
        "subcategory": "Uncategorized",
        "confidence": 0.0,
        "method": "uncategorized"
    }