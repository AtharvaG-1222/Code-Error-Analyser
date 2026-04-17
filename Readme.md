# CodAlytiX

CodAlytiX is a lightweight NLP-based classifier that identifies common programming errors and provides concise explanations.

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

## Model Details

Text preprocessing standardizes error messages by converting text to lowercase and removing special characters using regular expressions. This reduces noise and ensures similar error patterns are treated consistently.

TF-IDF converts text into numerical features by capturing important words and common phrases that indicate specific types of errors.

LinearSVC is used for classification because it performs well on high-dimensional text data and efficiently separates categories based on learned patterns.

The trained vectorizer and model are saved using joblib and reused during prediction for fast inference.
---

## Why Linear SVM Classifier (LinearSVC)
LinearSVC was chosen because TF-IDF produces high-dimensional and sparse text features, where linear decision boundaries are effective for separating categories based on keyword patterns.

Compared to Logistic Regression, LinearSVC is generally more robust for text classification tasks with large feature spaces and tends to provide better separation between classes when the distinction depends on specific terms or phrases. It also handles sparse data efficiently and trains quickly even when the number of features is high.

For short technical text such as programming error messages, LinearSVC provides a strong balance of performance, stability, and computational efficiency. 


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


## Working Of LLM

Set environment variable to enable explanations via HuggingFace API:
set HF_API_TOKEN=your_token

Else the application works without an API token using built-in fallback explanations, which have a set of predefined explainations for errors.

---

## Improvements Lined Up

- expand dataset with additional languages
- finer-grained error categories
- API-based deployment
- IDE integration