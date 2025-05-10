# 💻📄 **RAG Application on Agriculture Document-Based Retrieval**

A document retrieval system using Retrieval-Augmented Generation (RAG) for agriculture-related queries.

---
**🚀 Features**

**📄 PDF Processing**

Extracts text, images, and tables from uploaded PDFs.


Organizes extracted data into designated folders.


**🧠 Multi-modal Embeddings**


Uses Sentence Transformers for text embedding.


Uses CLIP model for image and text embeddings.


**🔍 Semantic Search**


Fast similarity search powered by FAISS for both image and text.


**🌐 Multilingual Support**


Supports English, Spanish, French, German, and Tamil using Google Translate API.


**💬 Conversational UI**


Chat-based interface styled with custom CSS.


Integrates Gemini (Google Generative AI) to answer questions about the documents.


**🎨 Beautiful Streamlit Interface**


Responsive design with elegant message bubbles.


Intuitive PDF uploader, language selector, and avatar-styled interaction.


**🧰 Tech Stack**


**Frontend/UI:** Streamlit + Custom CSS


**PDF Processing:** PyMuPDF (fitz)


**Text Embedding:** sentence-transformers (all-MiniLM-L6-v2)


**Image Embedding:** CLIPModel from HuggingFace Transformers


**Search Engine:** FAISS for vector similarity search


**Translation:** deep_translator (Google Translate)


**Generative AI:** Google Gemini API


**📁 Folder Structure**

project-root/
│____extracted
     ├── extracted_texts/       # Extracted text files
     ├── extracted_images/      # Extracted images from PDFs
     ├── extracted_tables/      # Placeholder for extracted tables
├── documentss_index.faiss # FAISS index for documents
├── metadatass.npy         # Metadata for text embeddings
├── image_indexss.faiss    # FAISS index for images
├── image_metadatass.npy   # Metadata for image embeddings
├── image_text_mapss.npy   # Mapping between image and corresponding text
└── rag_app.py 

**🌍 Supported Languages**
English (en)

Spanish (es)

French (fr)

German (de)

Tamil (ta)

**🧪 Future Enhancements**

🧾 Table Extraction and Embedding


📊 Visual Analytics of Document Content


🗂️ Document Classification and Tagging


🧠 Personalized Document Summarization

**🤝 Acknowledgements**

HuggingFace Transformers

PyMuPDF

Google Generative AI (Gemini)

Streamlit

Deep Translator

FAISS

**📜 License**

This project is open-source and available under the MIT License.

