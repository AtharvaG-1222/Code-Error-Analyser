import re
import joblib

model = joblib.load("model/classifier.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_error(error_text):
    cleaned = clean_text(error_text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


if __name__ == "__main__":
    samples = [
        "SyntaxError: invalid syntax on line 5",
        "unexpected token } in javascript",
        "list index out of range when looping",
        "ArrayIndexOutOfBoundsException java",
        "cant add string and number together",
        "cannot read property of undefined",
        "variable x is not defined",
        "cannot find symbol java",
        "no module named flask",
        "Error: Cannot find module express",
        "segmentation fault core dumped",
        "division by zero in average calculation",
        "null pointer exception java",
    ]
    for s in samples:
        print(f"{s:50s} -> {predict_error(s)}")
