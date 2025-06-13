import streamlit as st
import requests
import pdfplumber
import pandas as pd

HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b"

import os
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface(prompt):
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        # Check Content-Type before parsing
        if "application/json" not in response.headers.get("Content-Type", ""):
            return "⚠️ Error: Received non-JSON response. The model may still be loading or the API is overloaded."

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]['generated_text']
        else:
            return "⚠️ Error: Unexpected response format from the model."

    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP error {e.response.status_code}: {e.response.text}"
    except requests.exceptions.JSONDecodeError:
        return "⚠️ JSON decode error: Model response could not be parsed. Try again in 30 seconds."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


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
st.write(f"🔑 Token loaded: {HF_TOKEN[:8]}...")  # Just for debug

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

    # Handle API error messages gracefully
    if output.startswith("⚠️"):
        st.error(output)
    else:
        st.success("Flashcards generated!")
        try:
            qa_list = parse_flashcards(output)
            if not qa_list:
                st.warning("No valid Q&A pairs found.")
            else:
                for i, qa in enumerate(qa_list, 1):
                    st.markdown(f"**Q{i}:** {qa['Question']}")
                    st.markdown(f"**A{i}:** {qa['Answer']}")
                    st.markdown("---")

                df = pd.DataFrame(qa_list)
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv, "flashcards.csv", "text/csv")
        except Exception as e:
             st.error(f"Error while parsing flashcards: {str(e)}")

st.write(f"🔑 Token loaded: {HF_TOKEN[:8]}...")  # Just for debug
# so this is just to make sure it works fine also make suer it has all the above details of the new roman emipre 