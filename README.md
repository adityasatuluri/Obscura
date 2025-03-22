# Obscura AI

Obscura is an AI-powered redaction tool designed to detect and mask **Personally Identifiable Information (PII)** in text, documents, and images. It integrates **Named Entity Recognition (NER) models** with **Groq APIs** and **OCR technologies** to ensure high-accuracy data redaction.

## ✨ Features

- 🔍 **Advanced NER Model**: Fine-tuned entity recognition model with **96% accuracy**
- 📄 **Multi-format Support**: Process text, PDFs, Word documents, Excel sheets, and images
- 🖼️ **OCR Integration**: Redacts text from scanned documents and images using **PaddleOCR & Tesseract**
- 🔐 **Custom Sensitivity Levels**: Low, Medium, High, and Custom redaction modes
- 🌍 **Streamlit Web UI**: Simple and interactive frontend for easy document redaction

## 🛠️ Tech Stack

- **Python** (FastAPI, Streamlit)
- **PyTorch** (NER Model)
- **Groq API** (Generative Redaction Refinement)
- **PaddleOCR & Tesseract** (Text Extraction from Images)
- **Transformers** (Hugging Face Pipeline)
- **OpenCV & PyMuPDF** (Image & PDF Processing)

## 📋 Requirements

- Python 3.8+
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/adityasatuluri/Obscura.git
   cd Obscura
   ```

2. **Create a virtual environment and install dependencies**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the NER Model**:
   - Place the fine-tuned model inside `models/`

4. **Run the Streamlit Application**:
   ```sh
   streamlit run app.py
   ```

## 💻 Usage

### 🔹 Text Redaction
1. Enter text manually in the UI
2. Select the sensitivity level
3. Click "Mask Text" to redact sensitive information

### 🔹 File Redaction
1. Upload a PDF, DOCX, XLSX, TXT, or Image
2. The system processes the document and redacts sensitive entities
3. Download the redacted version

## 📊 Sensitivity Levels

| Level | Description |
|-------|-------------|
| Low | Redacts only the most critical PII (SSN, credit card numbers) |
| Medium | Redacts common PII (names, DOB, addresses) |
| High | Comprehensive redaction of all potential sensitive information |
| Custom | User-defined entity selection for targeted redaction |

## 📝 Examples

**Input Text:**
```
Patient John Smith (DOB: 04/12/1980) was admitted on 03/15/2023. 
His SSN is 123-45-6789 and he can be reached at john.smith@email.com or (555) 123-4567.
```

**Redacted Output:**
```
Patient [PERSON] (DOB: [DATE]) was admitted on [DATE]. 
His [SSN] is [REDACTED] and he can be reached at [EMAIL] or [PHONE].
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
