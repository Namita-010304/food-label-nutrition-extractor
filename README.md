# 🍎 Food Label Nutrition Extractor & Analyzer

> 📸 Scan any packaged food label using your camera, automatically extract nutrition facts using AI, and track your daily calories and sugar intake over time.

---

## ✨ **Motivation**

Modern lifestyles are surrounded by packaged foods, making it hard to keep track of daily calorie and sugar intake.  
⚡ This project aims to help people:
- Quickly **understand what they're eating**
- Detect **high sugar, salt, or allergens**
- Track and **visualize nutrition trends** over time
- Make **healthier choices** with recommendations

By combining **Computer Vision, OCR, NLP, and Data Science**, it transforms raw food labels into actionable health insights.

---

## 🧠 **How it works**

✅ **Camera or Image Input**  
User points their phone or webcam at a food label.

✅ **Image Preprocessing**  
Enhances image quality using OpenCV (grayscale, sharpening, thresholding).

✅ **OCR (Optical Character Recognition)**  
Uses deep learning OCR (EasyOCR, TrOCR) to extract text from the label.

✅ **NLP & Parsing**  
- Parses text to extract numeric values like calories, sugar, protein, fat.
- Detects keywords for allergens (e.g., peanuts, gluten).

✅ **Data Storage & Analysis**  
- Stores scanned data locally (e.g., SQLite, CSV, JSON).
- Visualizes daily, weekly, monthly nutrition trends.
- Optionally compares against daily recommended limits.

✅ **Optional AI Extensions**  
- Time series forecasting: predict weekly calorie trends.
- Clustering: group foods by health risk.
- Recommender: suggest lower-sugar alternatives.

---

## 📊 **Features**

- 📸 Scan and extract nutrition facts in real time
- 🗣 Voice summary: "This product contains 25g sugar, which is above your daily limit."
- 📈 Charts: visualize weekly sugar and calorie intake
- 🧾 History: see previously scanned items
- ⚠ Allergen detection: alerts if allergens found
- 🧠 Smart recommendations: suggest healthier products (planned)

---

## 🛠 **Tech Stack & ML / AI Alignment**

| Layer | What | Why it matters |
|--|--|--|
| Computer Vision | OpenCV | Preprocess images for better OCR |
| OCR (AI) | EasyOCR, TrOCR | Deep learning models to read text |
| NLP | Python regex / spaCy | Parse and extract nutrition fields |
| Data Storage | CSV / SQLite | Keep history for analysis |
| Data Science | pandas, matplotlib | Visualization & trend analysis |
| Voice | pyttsx3 / gTTS | Text-to-speech feedback |

---


