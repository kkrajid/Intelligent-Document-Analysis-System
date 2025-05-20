# Intelligent Document Analysis System

## Overview
The **Intelligent Document Analysis System** is a Streamlit-based web application that enables users to upload documents (PDF, DOCX, TXT), extract text, and query content using advanced AI capabilities powered by Google's Gemini models and LangChain's Retrieval-Augmented Generation (RAG) pipeline. The system processes documents, creates a FAISS vector store for efficient retrieval, and provides structured responses with source references and confidence scores.

## Features
- **Multi-Format Support**: Upload and process PDF, DOCX, and TXT files (up to 25MB per file).
- **Text Extraction**: Extracts text and metadata (e.g., source, page numbers) from supported file types.
- **AI-Powered Q&A**: Query documents using natural language, with responses including direct answers, evidence, source references, and confidence estimates.
- **Vector Store**: Uses FAISS and Google Generative AI embeddings for fast and accurate document retrieval.
- **Interactive UI**: Streamlit interface with chat history, document analytics, and model configuration (model selection, creativity level).
- **Caching**: Implements LangChain's SQLiteCache and Streamlit's `@st.cache_resource` for performance optimization.
- **Export Capabilities**: Export chat history as CSV for record-keeping.
- **Document Analytics**: Displays file metadata (e.g., page count) and previews in the sidebar.

## Prerequisites
- Python 3.8+
- A Google API key for Gemini models (set in a `.env` file).
- Tesseract OCR (optional, for scanned PDF support with additional setup).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Create a virtual environment and install required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install streamlit PyPDF2 langchain langchain-google-genai faiss-cpu python-dotenv pandas python-docx
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Google API key:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **(Optional) Install OCR Support**:
   For scanned PDFs, install Tesseract OCR and additional Python packages:
   ```bash
   pip install pytesseract pdf2image
   ```
   Install Tesseract OCR:
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download and install from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - **macOS**: `brew install tesseract`

## Usage
1. **Run the Application**:
   Start the Streamlit server:
   ```bash
   streamlit run chatpdf1.py
   ```
   Replace `chatpdf1.py` with the name of your Python script containing the provided code.

2. **Upload Documents**:
   - Navigate to the sidebar and upload PDF, DOCX, or TXT files (max 25MB each).
   - Click "Process Documents" to extract text and create a vector store.

3. **Query Documents**:
   - Enter questions in the chat input box (e.g., "What is the main topic of the document?").
   - View responses with direct answers, evidence, source references, and confidence scores.
   - Expand responses to see detailed source information.

4. **Configure AI**:
   - In the sidebar, select a Gemini model (e.g., `gemini-1.5-flash-latest`) and adjust the creativity level (temperature) via the slider.

5. **Analytics and Export**:
   - View document metadata (e.g., file name, page count) in the sidebar.
   - Download chat history as a CSV file for record-keeping.

## Configuration
- **Supported File Types**: PDF, DOCX, TXT.
- **File Size Limit**: 25MB per file.
- **AI Models**: Choose from `gemini-1.5-flash-latest` or `gemini-1.0-pro-latest`.
- **Temperature**: Adjust creativity level (0.0 to 1.0, default 0.3) for response generation.
- **Chunking**: Text is split into 2000-character chunks with 200-character overlap.
- **Retrieval**: Retrieves top 5 relevant chunks for each query (configurable in code).

## Limitations
- **Scanned PDFs**: Requires Tesseract OCR for image-based PDFs (not included in the default code).
- **Metadata Handling**: Multi-file uploads may lose specific source attribution for chunks (see improvements below).
- **API Dependency**: Requires a valid Google API key and internet access for Gemini models.

## Potential Improvements
- **OCR Support**: Add `pytesseract` and `pdf2image` for scanned PDFs.
- **Improved Metadata**: Map chunks to specific source files and pages for multi-file uploads.
- **Configurable Parameters**: Allow users to adjust chunk size and retrieval `k` via the UI.
- **JSON Export**: Add support for exporting chat history as JSON.
- **Error Handling**: Enhance error messages for specific failure cases (e.g., API quota exceeded).
- **Performance**: Implement batch processing for embeddings to reduce API calls.

## Troubleshooting
- **API Errors**: Ensure your Google API key is valid and has sufficient quota. Check `.env` file setup.
- **File Processing Errors**: Verify files are not corrupted and meet size/type requirements.
- **Vector Store Issues**: Clear the cache using the "Clear Cache" button if document processing fails.
- **UI Issues**: Ensure `streamlit run` is executed from the correct directory with dependencies installed.

## Contributing
Contributions are welcome! Please submit pull requests or open issues for bug reports, feature requests, or improvements.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, contact [your-email@example.com] or open an issue on the repository.
