




# import os
# from dotenv import load_dotenv
# import google.generativeai as genai
# import base64
# import cv2
# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# import pyttsx3
# import numpy as np
# import difflib

# # Load API key from .env file
# load_dotenv()
# api_key = os.environ.get("GEMINI_API_KEY")
# genai.configure(api_key=api_key)

# # Corrections for common OCR mistakes
# OCR_CORRECTIONS = {
#     'contalner': 'container',
#     'protzels': 'pretzels',
#     'sodlum': 'sodium',
#     'calclum': 'calcium',
#     'calorles': 'calories',
#     'servlng': 'serving',
#     'slze': 'size',
#     'prote-n': 'protein',
#     # add more if needed
# }

# def correct_ocr_text(text):
#     words = text.lower().split()
#     corrected = []
#     for word in words:
#         if word in OCR_CORRECTIONS:
#             corrected.append(OCR_CORRECTIONS[word])
#         else:
#             # Try to find similar word
#             match = difflib.get_close_matches(word, OCR_CORRECTIONS.keys(), n=1, cutoff=0.8)
#             if match:
#                 corrected.append(OCR_CORRECTIONS[match[0]])
#             else:
#                 corrected.append(word)
#     return ' '.join(corrected)

# def preprocess_image(image_path):
#     # Load image in grayscale
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at: {image_path}")
    
#     # Deskew
#     coords = np.column_stack(np.where(image < 255))
#     angle = cv2.minAreaRect(coords)[-1]
#     angle = -(90 + angle) if angle < -45 else -angle
#     (h, w) = image.shape
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     # Enhance
#     blurred = cv2.GaussianBlur(deskewed, (3, 3), 0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blurred)
#     thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     return thresh

# def encode_image_to_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode('utf-8')
         
# def extract_text_with_gemini(image_path):
#     with open(image_path, "rb") as f:
#         image_bytes = f.read()

#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content([
#         "Extract all visible text from this nutrition label image.",
#         {
#             "inline_data": {
#                 "mime_type": "image/png",  # or image/jpeg
#                 "data": image_bytes
#             }
#         }
#     ])
#     text = response.text.strip()
#     corrected = correct_ocr_text(text)
#     return corrected

# def extract_nutrition(text):
#     print("Parsing nutrition data...")
#     data = {}
#     patterns = {
#         'calories': r'calories\s*(\d+)',
#         'total_fat': r'total fat\s*(\d+g?)',
#         'saturated_fat': r'saturated fat\s*(\d+g?)',
#         'trans_fat': r'trans fat\s*(\d+g?)',
#         'cholesterol': r'cholesterol\s*(\d+mg?)',
#         'sodium': r'sodium\s*(\d+mg?)',
#         'total_carbohydrate': r'total carbohydrate\s*(\d+g?)',
#         'dietary_fiber': r'dietary fiber\s*(\d+g?)',
#         'total_sugars': r'total sugars\s*(\d+g?)',
#         'protein': r'protein\s*(\d+g?)',
#         'serving_size': r'serving size\s*([^\n]+)',
#     }
#     for key, pattern in patterns.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         data[key] = match.group(1).strip() if match else 'N/A'
#     return data

# def store_data(nutrition_data):
#     print("Saving data and creating chart...")
#     df = pd.DataFrame([nutrition_data])
#     file_exists = os.path.exists('nutrition_data.csv')
#     df.to_csv('nutrition_data.csv', index=False, mode='a', header=not file_exists)

#     # Create a bar chart
#     plot_data = {k: v for k, v in nutrition_data.items() if v != 'N/A' and k != 'serving_size'}
#     numeric = {k: float(re.sub(r'[a-zA-Z]', '', v)) for k, v in plot_data.items()}
#     plt.figure(figsize=(10, 6))
#     plt.bar(numeric.keys(), numeric.values(), color='skyblue')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig('nutrition_chart.png')
#     print("Chart saved as nutrition_chart.png")

# def voice_summary(nutrition_data):
#     print("Reading summary...")
#     engine = pyttsx3.init()
#     summary = f"Calories: {nutrition_data.get('calories')}. Protein: {nutrition_data.get('protein')}. Total sugars: {nutrition_data.get('total_sugars')}."
#     engine.say(summary)
#     engine.runAndWait()

# def main(image_path):
#     if not os.path.exists(image_path):
#         print(f"File not found: {image_path}")
#         return
#     processed = preprocess_image(image_path)
#     temp_image = "temp_processed.png"
#     cv2.imwrite(temp_image, processed)

#     try:
#         text = extract_text_with_gemini(temp_image)
#         print("\n--- Extracted Text ---\n", text)

#         nutrition = extract_nutrition(text)
#         print("\n--- Nutrition Data ---\n", nutrition)

#         if any(v != 'N/A' for v in nutrition.values()):
#             store_data(nutrition)
#             voice_summary(nutrition)
#         else:
#             print("Could not extract nutrition data.")
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         if os.path.exists(temp_image):
#             os.remove(temp_image)

# if __name__ == "__main__":
#     image_path = 'data/food_nutrition_label.png'  # Change if your image is in another location
#     main(image_path)









import os
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import cv2
import re
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
import numpy as np
import difflib

# Load env
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

OCR_CORRECTIONS = {
    'contalner': 'container',
    'protzels': 'pretzels',
    'sodlum': 'sodium',
    'calclum': 'calcium',
    'calorles': 'calories',
    'servlng': 'serving',
    'slze': 'size',
    'prote-n': 'protein',
}

def correct_ocr_text(text):
    words = text.lower().split()
    corrected = []
    for word in words:
        if word in OCR_CORRECTIONS:
            corrected.append(OCR_CORRECTIONS[word])
        else:
            matches = difflib.get_close_matches(word, OCR_CORRECTIONS.keys(), n=1, cutoff=0.8)
            corrected.append(OCR_CORRECTIONS[matches[0]] if matches else word)
    return ' '.join(corrected)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coords = np.column_stack(np.where(image < 255))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(deskewed)
    return enhanced

def extract_text_with_gemini(image_path):
    print("Extracting text with Gemini...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        "Extract all visible nutrition text from this label image.",
        {
            "inline_data": {
                "mime_type": "image/png",
                "data": image_bytes
            }
        }
    ])
    text = response.text.strip()
    return correct_ocr_text(text)

def extract_nutrition(text):
    print("Parsing nutrition data...")
    data = {}
    patterns = {
    'calories': r'calories[^\d]*(\d+)',
    'total_fat': r'total fat[^\d]*(\d+\.?\d*g?)',
    'saturated_fat': r'saturated fat[^\d]*(\d+\.?\d*g?)',
    'trans_fat': r'trans fat[^\d]*(\d+\.?\d*g?)',
    'cholesterol': r'cholesterol[^\d]*(\d+\.?\d*mg?)',
    'sodium': r'sodium[^\d]*(\d+\.?\d*mg?)',
    'total_carbohydrate': r'total carb.*?[^\d]*(\d+\.?\d*g?)',
    'dietary_fiber': r'dietary fiber[^\d]*(\d+\.?\d*g?)',
    'total_sugars': r'total sugars[^\d]*(\d+\.?\d*g?)',
    'protein': r'protein[^\d]*(\d+\.?\d*g?)',
    'vitamin d': r'vitamin d[^\d]*(\d+\.?\d*mcg?)',
    'calcium': r'calcium[^\d]*(\d+\.?\d*mg?)',
    'iron': r'iron[^\d]*(\d+\.?\d*mg?)',
}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        data[key] = match.group(1).strip() if match else 'N/A'
    return data

def store_data(nutrition_data):
    df = pd.DataFrame([nutrition_data])
    df.to_csv('nutrition_data.csv', index=False, mode='a', header=not os.path.exists('nutrition_data.csv'))
    print("Saved to nutrition_data.csv")
    plot_data = {k: v for k,v in nutrition_data.items() if v!='N/A' and k!='serving_size'}
    numeric = {k: float(re.sub(r'[a-zA-Z<]', '', v)) for k,v in plot_data.items()}
    plt.figure(figsize=(10,6))
    plt.bar(numeric.keys(), numeric.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('nutrition_chart.png')
    print("Chart saved to nutrition_chart.png")

def voice_summary(nutrition_data):
    engine = pyttsx3.init()
    summary = f"Calories: {nutrition_data.get('calories')}. Protein: {nutrition_data.get('protein')}. Total Sugars: {nutrition_data.get('total_sugars')}."
    print("Speaking summary...")
    engine.say(summary)
    engine.runAndWait()

def plot_macro_micro(nutrition_data):
    print("Plotting macro and micronutrients in a single figure...")

    # Define keys
    macro_keys = ['calories', 'total_fat', 'saturated_fat', 'trans_fat',
                  'cholesterol', 'sodium', 'total_carbohydrate', 'dietary_fiber',
                  'total_sugars', 'protein']
    micro_keys = ['vitamin d', 'calcium', 'iron']

    # Filter and clean macro data
    macro_data = {}
    for k in macro_keys:
        v = nutrition_data.get(k)
        if v and v != 'N/A':
            try:
                macro_data[k] = float(re.sub(r'[^\d\.]', '', v))
            except:
                continue

    # Filter and clean micro data
    micro_data = {}
    for k in micro_keys:
        v = nutrition_data.get(k)
        if v and v != 'N/A':
            try:
                micro_data[k] = float(re.sub(r'[^\d\.]', '', v))
            except:
                continue

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot macros
    if macro_data:
        axes[0].bar(macro_data.keys(), macro_data.values(), color='cornflowerblue')
        axes[0].set_title('Macronutrients')
        axes[0].set_ylabel('Amount')
        axes[0].tick_params(axis='x', rotation=45)
    else:
        axes[0].text(0.5, 0.5, 'No macronutrient data', ha='center', va='center')
        axes[0].set_title('Macronutrients')

    # Plot micros
    if micro_data:
        axes[1].bar(micro_data.keys(), micro_data.values(), color='mediumseagreen')
        axes[1].set_title('Micronutrients')
        axes[1].set_ylabel('Amount')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No micronutrient data', ha='center', va='center')
        axes[1].set_title('Micronutrients')

    plt.tight_layout()
    plt.savefig('nutrients_combined_chart.png')
    print("Combined chart saved to nutrients_combined_chart.png")


def main(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    processed = preprocess_image(image_path)
    temp = "temp_processed.png"
    cv2.imwrite(temp, processed)
    try:
        text = extract_text_with_gemini(temp)
        print("\n--- Extracted Text ---\n", text)
        nutrition = extract_nutrition(text)
        print("\n--- Nutrition Data ---\n", nutrition)
        if any(v != 'N/A' for v in nutrition.values()):
            store_data(nutrition)
            plot_macro_micro(nutrition)
            voice_summary(nutrition)
        else:
            print("Could not extract nutrition data.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(temp):
            os.remove(temp)

if __name__ == "__main__":
    image_path = 'data/nutrition.png'
    main(image_path)
