import streamlit as st
import requests
import pdfplumber
import pandas as pd

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = "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"  # 🔁 Replace this with your Hugging Face token
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface(prompt):
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()
    try:
        return result[0]['generated_text']
    except:
        return "Error generating flashcards. Try again later."

def extract_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def parse_flashcards(raw_output):
    lines = raw_output.strip().split('\n')
    qa_pairs = []
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            qa_pairs.append({"Question": lines[i], "Answer": lines[i+1]})
    return qa_pairs

st.title("📚 LLM-Powered Flashcard Generator")

input_method = st.radio("Select input method:", ["Paste Text", "Upload PDF"])

input_text = ""
if input_method == "Paste Text":
    input_text = st.text_area("Paste your educational content here:")
elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        input_text = extract_pdf_text(uploaded_file)

if st.button("Generate Flashcards") and input_text.strip():
    with st.spinner("Generating flashcards..."):
        prompt = f"Generate 10 Q&A flashcards based on the following content:\n\n{input_text}"
        output = query_huggingface(prompt)
        st.success("Flashcards generated!")

        qa_list = parse_flashcards(output)
        for i, qa in enumerate(qa_list, 1):
            st.markdown(f"**Q{i}:** {qa['Question']}")
            st.markdown(f"**A{i}:** {qa['Answer']}")
            st.markdown("---")

        df = pd.DataFrame(qa_list)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "flashcards.csv", "text/csv")
