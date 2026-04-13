import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

# Load Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Helper Functions
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page_text := page.extract_text():
                text += page_text + "\n"
    return text.strip()

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def truncate_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

def summarize_text(text, detail_level="Medium"):
    length_map = {
        "Short": (80, 30),
        "Medium": (150, 60),
        "Detailed": (250, 100)
    }
    max_len, min_len = length_map.get(detail_level, (150, 60))
    text = truncate_text(text, tokenizer)
    return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

def ask_question(context, question):
    return qa_pipeline(question=question, context=context)["answer"]

# Streamlit Interface
st.title("📄 Text & PDF Summarizer with Q&A")

# File uploader and input options
input_type = st.radio("Choose your input type:", ("Upload PDF", "Upload .txt", "Paste Plain Text"))
text_content = ""

if input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        text_content = extract_text_from_pdf(pdf_file)

elif input_type == "Upload .txt":
    txt_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if txt_file:
        text_content = extract_text_from_txt(txt_file)

elif input_type == "Paste Plain Text":
    text_content = st.text_area("Paste your text here:")

# Summary detail level
detail_level = st.selectbox("Select summary detail level:", ["Short", "Medium", "Detailed"])

# Summarize button
if text_content and st.button("Summarize Text"):
    with st.spinner("Generating summary..."):
        summary = summarize_text(text_content, detail_level)
    st.subheader("📝 Summary")
    st.write(summary)
    st.session_state["summary"] = summary
    st.session_state["context"] = text_content

# Q&A
if "summary" in st.session_state and "context" in st.session_state:
    st.subheader("❓ Ask Questions About the Summary")
    question = st.text_input("Enter your question")
    if question:
        with st.spinner("Getting answer..."):
            answer = ask_question(st.session_state["context"], question)
        st.write("💡 Answer:")
        st.write(answer)
