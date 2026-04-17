import streamlit as st
import re
import joblib
from explain import explain_error

st.set_page_config(page_title="Error Insight", layout="centered")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("model/classifier.joblib")
    vectorizer = joblib.load("model/vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_model()

st.title("Error Insight")
st.markdown("Classify and explain programming errors across multiple languages.")

error_text = st.text_area("Paste error message", height=100,
    placeholder="e.g. IndexError: list index out of range")

if st.button("Analyze", type="primary"):
    if not error_text.strip():
        st.warning("Enter an error message first.")
    else:
        cleaned = clean_text(error_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        st.subheader("Error type")
        st.write(prediction)

        with st.spinner("generating explanation..."):
            explanation = explain_error(error_text, prediction)

        st.subheader("Explanation")
        st.markdown(explanation)
