# Error Insight

Error Insight is a lightweight NLP-based classifier that identifies common programming errors and provides concise explanations.

The project combines traditional machine learning with an optional language model API to help quickly interpret unfamiliar error messages across different programming languages.

---

## Key Features

- Classifies common programming errors using NLP
- Works across multiple languages (Python, JavaScript, Java, C++, etc.)
- Generates short, practical explanations
- Fast local inference using a trained model
- Simple web interface for quick testing

---

## Error Categories

- Syntax Error
- Type Error
- Name Error
- Index Error
- Import Error
- Runtime Error

---

## How it Works

1. error message is cleaned and normalized
2. TF-IDF converts text into numerical features
3. Linear SVM model predicts error category
4. explanation is generated using LLM API or fallback logic
5. result is displayed via Streamlit interface

---

## Tech Stack

- scikit-learn (LinearSVC, TF-IDF)
- Streamlit
- HuggingFace Inference API (optional)
- joblib

---

## Project Structure
│
├── data/
│ └── error_messages.csv
│
├── model/
│ ├── classifier.joblib
│ └── vectorizer.joblib
│
├── train.py
├── predict.py
├── explain.py
├── app.py
└── requirements.txt


---
## Setup

Install dependencies:
pip install -r requirements.txt

Train model:
python train.py

Run application:
streamlit run app.py
---

## Model Details

Text preprocessing removes noise and standardizes input using regex.

TF-IDF is used to capture important keywords and phrases commonly associated with specific types of programming errors.

LinearSVC was chosen because it performs well on sparse text features and provides fast, stable classification.

The trained model is saved locally and reused for prediction.

---

## Optional LLM explanations

Set environment variable to enable explanations via HuggingFace API:
set HF_API_TOKEN=your_token
The application works without an API token using built-in fallback explanations.

---

## Possible Improvements

- expand dataset with additional languages
- finer-grained error categories
- API-based deployment
- IDE integration