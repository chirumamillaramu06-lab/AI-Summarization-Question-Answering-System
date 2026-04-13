# 📄 AI-Powered Document Summarisation and Question Answering App

This Streamlit app enables users to upload a **PDF**, **.txt** file, or paste plain text to generate AI-powered summaries and ask questions based on the content. It uses state-of-the-art NLP models from Hugging Face Transformers — **BART** for summarisation and **RoBERTa** for question answering.

---

## 🚀 Features

- 📤 Upload PDF or TXT files
- 📝 Paste plain text directly
- 🧠 Summarise documents using `facebook/bart-large-cnn`
- ❓ Ask questions using `deepset/roberta-base-squad2`
- 🔁 Choose summary detail level: Short / Medium / Detailed
- ⚡ Real-time response using Hugging Face pipelines

---

## 🧰 Technologies Used

- [Streamlit](https://streamlit.io/) — Interactive UI
- [pdfplumber](https://github.com/jsvine/pdfplumber) — Extracts text from PDFs
- [Transformers (Hugging Face)](https://huggingface.co/transformers)
  - `facebook/bart-large-cnn` — Summarisation model
  - `deepset/roberta-base-squad2` — Q&A model
- [PyTorch](https://pytorch.org/) — Backend for NLP models

---

## 📦 Installation Guide

### 🔧 Step 1: Create a Virtual Environment (Optional but Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate


### 🔧 Step 2: Create a Virtual Environment (Optional but Recommended)

pip install -r requirements.txt


### 🔧 Step 3: ▶️ Running the App

streamlit run app.py

### Once the app launches, open your browser and go to:
http://localhost:8501







