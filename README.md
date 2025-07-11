# 📄 Ammy DocAI – Your Document Assistant Powered by AI

🚀 Live Demo: [ammy-docai.streamlit.app](https://ammy-docai.streamlit.app/)

Ammy DocAI is a lightweight and intelligent Streamlit application that allows you to upload documents and ask questions about their content. It uses powerful language models via Cohere API to understand and respond to natural language queries based on your uploaded data.

---

## 🧠 Features

- 📁 Upload PDFs or text files
- 💬 Ask questions about the content
- ⚡ Fast and contextual answers using Cohere
- 🌙 Dark mode Streamlit UI
- 🔐 Secrets management for API keys

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – for the UI and app deployment
- [Cohere](https://cohere.com/) – for natural language processing
- [Python](https://www.python.org/) – core language
- [PyPDF2 / pdfplumber] – for PDF parsing
- [dotenv / streamlit.secrets] – for managing API keys

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ammy-docai.git
cd ammy-docai
pip install -r requirements.txt
```
## 🚀 Run Locally

```bash
streamlit run app.py
```
## 🚩 Demo

Try the live app here:  
https://ammy-docai.streamlit.app/

---

## 🧾 Example Use Case

> Upload a PDF (e.g., legal document) and ask:
>
> _"What is the refund policy mentioned in this document?"_

---

## 📁 Folder Structure

``# 
ammy-docai/
├── app.py
├── requirements.txt
├── utils.py
├── .streamlit/
│   └── secrets.toml (ignored in Git)
└── README.md
``# 

---

## 🔐 Setup

1. Create a `.streamlit/secrets.toml` file:
   ```toml
   COHERE_API_KEY = "your-cohere-api-key"
   ```

2. Or, for local development, use a `.env` file:
   ```env
   COHERE_API_KEY=your-cohere-api-key
   ```

---

## 🚀 Run Locally

```bash
streamlit run app.py
``` 

---

## 🛡️ Security

- API keys are managed via `secrets.toml` or Streamlit Cloud Secrets UI.
- Sensitive files are added to `.gitignore` and never pushed to GitHub.

---

## 🧠 Tips

- Ensure your documents are readable PDFs (not scanned images).
- For best results, use concise and clear questions.
- Can be extended to support DOCX, OCR, or multilingual support.

---

## 📄 License

MIT License. Feel free to fork, improve, or build upon this project.

---

## 🙋‍♀️ Author

Built with ❤️ by [Amisha Sharma](https://github.com/ammycodex)
