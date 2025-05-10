import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from streamlit_option_menu import option_menu
import time

# Define supported languages
supported_languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Tamil": "ta"
}


# Configure Gemini API key
genai.configure(api_key="api_key")  # Replace with your actual Gemini API key  
chat = genai.GenerativeModel("models/gemini-2.0-flash")

# Set base directory
BASE_DIR = os.getcwd()
TEXT_FOLDER = os.path.join(BASE_DIR, "extracted_texts")
IMAGE_FOLDER = os.path.join(BASE_DIR, "extracted_images")
TABLE_FOLDER = os.path.join(BASE_DIR, "extracted_tables")

DOC_INDEX_PATH = os.path.join(BASE_DIR, "documentss_index.faiss")
DOC_META_PATH = os.path.join(BASE_DIR, "metadatass.npy")
IMG_INDEX_PATH = os.path.join(BASE_DIR, "image_indexss.faiss")
IMG_META_PATH = os.path.join(BASE_DIR, "image_metadatass.npy")
IMAGE_TEXT_MAPPING_PATH = os.path.join(BASE_DIR, "image_text_mapss.npy")

# Ensure directories exist
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(TABLE_FOLDER, exist_ok=True)

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# PDF Processing Functions
def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(i)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_folder, f"page_{i+1}_img_{img_index + 1}.{image_ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)
    return image_paths

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# Embedding Functions
def get_image_embeddings(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs).cpu().numpy()
    return embeddings

def get_text_embeddings(texts):
    return text_model.encode(texts)

def get_clip_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs).cpu().numpy()
    return text_features

def t(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text


# ======================================
# CUSTOM CHAT UI COMPONENTS
# ======================================
# ======================================
# CUSTOM CHAT UI COMPONENTS
# ======================================
st.markdown("""
<style>
    /* Main chat container */
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
    }
    
    /* Message bubbles */
    .message {
        padding: 12px 16px;
        margin-bottom: 16px;
        border-radius: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: flex-start;
        line-height: 1.5;
        position: relative;
        animation: fadeIn 0.3s ease-out;
    }
    
    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border-radius: 18px 18px 0 18px;
        margin-left: auto;
        max-width: 75%;
        margin-right: 8px;
    }
    
    /* AI message bubble */
    .ai-message {
        background: linear-gradient(135deg, #f1f1f1, #e6e6e6);
        color: #333;
        border-radius: 18px 18px 18px 0;
        margin-right: auto;
        max-width: 75%;
        margin-left: 8px;
    }
    
    /* Avatar styling */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin-right: 12px;
        object-fit: cover;
        flex-shrink: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Typing indicator */
    .typing {
        display: inline-flex;
        padding: 12px 16px;
        background: #f1f1f1;
        border-radius: 18px 18px 18px 0;
        align-items: center;
    }
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #666;
        margin: 0 3px;
        animation: typing-animation 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing-animation {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50 !important;
        border-radius: 12px !important;
        padding: 30px !important;
        background: rgba(76, 175, 80, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(76, 175, 80, 0.1) !important;
        border-color: #3d8b40 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef) !important;
    }
    
    /* Status messages */
    .stStatus {
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    /* Image results */
    .image-result {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    
    .image-result:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .user-message, .ai-message {
            max-width: 85%;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3d8b40;
    }
</style>
""", unsafe_allow_html=True)

# Avatar images (using modern SVG icons)
#USER_AVATAR = https://cdn-icons-png.flaticon.com/512/1144/1144760.png
#AI_AVATAR = https://cdn-icons-png.flaticon.com/512/4712/4712109.png  # Modern AI icon

# ======================================
# CHAT UI HELPER FUNCTIONS
# ======================================
def user_message(text):
    st.markdown(f"""
    <div class="message user-message">
        <span>{text}</span>
        <img src="https://cdn-icons-png.flaticon.com/512/1144/1144760.png" class="avatar" style="margin-left: 12px; margin-right: 0;">
    </div>
    """, unsafe_allow_html=True)


def ai_message(text):
    # Convert **bold** to <b> tags
    import re
    processed_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Replace newlines with <br> for HTML rendering
    processed_text = processed_text.replace('\n', '<br>')

    st.markdown(f"""
        <style>
        .ai-message {{
            display: flex;
            align-items: flex-start;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
            color: #333;
        }}
        .ai-message .avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 15px;
        }}
        .ai-message .message-text {{
            line-height: 1.6;
        }}
        </style>
        <div class="ai-message">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="avatar">
            <div class="message-text">{processed_text}</div>
        </div>
    """, unsafe_allow_html=True)



def typing_indicator():
    st.markdown("""
    <div class="typing">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.5)  # Simulate thinking time

def display_image(image_path):
    """Display an image in a styled container (simpler version)"""
    st.markdown("""<div class="image-result">""", unsafe_allow_html=True)
    st.image(image_path, use_column_width=True)
    st.markdown("""</div>""", unsafe_allow_html=True)

# ======================================
# ENHANCED UI COMPONENTS
# ======================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="color: #4CAF50;">🌐 Language Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    selected_language = st.selectbox(
        "Choose your language", 
        list(supported_languages.keys()),
        label_visibility="collapsed"
    )
    lang_code = supported_languages[selected_language]
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <small>Document Assistant v1.0</small><br>
        <small>Upload PDFs and chat with their contents</small>
    </div>
    """, unsafe_allow_html=True)

# Main chat header
st.markdown(f"""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #4CAF50; font-weight: 700;">📄🔍 Agri - Document Assistant</h1>
    <p style="color: #666; font-size: 1.1rem;">Upload a PDF and have a conversation with your documents</p>
</div>
""", unsafe_allow_html=True)

# Enhanced file upload section
upload_col1, upload_col2, upload_col3 = st.columns([1,8,1])
with upload_col2:
    with st.expander("📤 Upload Document", expanded=True):
        uploaded_file = st.file_uploader(
            t("Upload a PDF document", lang_code), 
            type=["pdf"],
            help="Supported formats: PDF",
            label_visibility="collapsed",
            accept_multiple_files=False
        )

        if uploaded_file:
            with st.spinner("Processing document..."):
                # ... [YOUR ORIGINAL FILE PROCESSING CODE] ...
                pdf_path = os.path.join(BASE_DIR, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Save extracted text
                text = extract_text_from_pdf(pdf_path)
                text_file_path = os.path.join(TEXT_FOLDER, uploaded_file.name + ".txt")
                with open(text_file_path, "w", encoding="utf-8") as f:
                    f.write(text)

                # Save extracted images
                image_output_dir = os.path.join(IMAGE_FOLDER, uploaded_file.name)
                os.makedirs(image_output_dir, exist_ok=True)
                image_paths = extract_images_from_pdf(pdf_path, image_output_dir)
                
                st.success(t("✅ Document processed successfully!", lang_code))
                st.balloons()

# Indexing section with better visual feedback
if DOC_INDEX_PATH is None or IMG_INDEX_PATH is None:
    col1, col2, col3 = st.columns([3,4,3])
    with col2:
        if st.button("🔎 Create Search Index", use_container_width=True):
            with st.status("🔨 Creating search index...", expanded=True) as status:
                # ... [INDEXING CODE] ...
                text_files = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]
                texts = []
                doc_file_names = []

                for file in text_files:
                    try:
                        with open(os.path.join(TEXT_FOLDER, file), "r", encoding="utf-8") as f:
                            texts.append(f.read())
                            doc_file_names.append(file)
                    except:
                        continue

                if texts:
                    st.write(t("📝 Processing text content...",lang_code))
                    text_embeddings = get_text_embeddings(texts)
                    doc_index = faiss.IndexFlatL2(text_embeddings.shape[1])
                    doc_index.add(np.array(text_embeddings))
                    faiss.write_index(doc_index, DOC_INDEX_PATH)
                    np.save(DOC_META_PATH, doc_file_names)

                # Image embeddings
                st.write(t("🖼️ Processing images...", lang_code))
                image_paths = []
                for folder in os.listdir(IMAGE_FOLDER):
                    folder_path = os.path.join(IMAGE_FOLDER, folder)
                    if os.path.isdir(folder_path):
                        for file in os.listdir(folder_path):
                            image_paths.append(os.path.join(folder_path, file))

                if image_paths:
                    image_embeddings = get_image_embeddings(image_paths)
                    image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
                    image_index.add(image_embeddings)
                    faiss.write_index(image_index, IMG_INDEX_PATH)
                    np.save(IMG_META_PATH, image_paths)

                    # Generate image-to-text mapping using CLIP similarity
                    img_text_map = {}
                    if image_paths and texts:
                        progress_bar = st.progress(0)
                        total_images = len(image_paths)
                        
                        for i, (img_path, img_embed) in enumerate(zip(image_paths, image_embeddings)):
                            max_sim = -1
                            best_para = ""
                            for doc_text in texts:
                                for para in doc_text.split('\n\n'):
                                    para = para.strip()
                                    if not para:
                                        continue
                                    para_embed = get_clip_text_embedding(para)[0]
                                    sim = cosine_similarity([img_embed], [para_embed])[0][0]
                                    if sim > max_sim:
                                        max_sim = sim
                                        best_para = para
                            # Use folder::filename as key
                            img_key = os.path.join(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))
                            img_text_map[img_key] = best_para
                            progress_bar.progress((i + 1) / total_images)

                        np.save(IMAGE_TEXT_MAPPING_PATH, img_text_map)
                
                status.update(label="🎉 Index created successfully!", state="complete")
                st.success(t("You can now ask questions about your documents!", lang_code))

# Enhanced chat interface
query = st.chat_input(t("Ask a question about your documents...", lang_code))

if query:
    user_message(query)
    image_keywords = ["image", "figure", "chart", "graph", "diagram", "illustration", "photo", "visual", "picture"]
    show_images = any(keyword in query.lower() for keyword in image_keywords)
    
    with st.spinner("Analyzing your question..."):
        #typing_indicator()
        
        try:
            # ... [SEARCH CODE] ...
            doc_index = faiss.read_index(DOC_INDEX_PATH)
            doc_file_names = np.load(DOC_META_PATH, allow_pickle=True)
            query_embedding = get_text_embeddings([query])
            D, I = doc_index.search(np.array(query_embedding), k=3)

            retrieved_contents = []

            for idx in I[0]:
                if idx < len(doc_file_names):
                    filename = doc_file_names[idx]
                    text_path = os.path.join(TEXT_FOLDER, filename)
                    try:
                        with open(text_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            retrieved_contents.append(content)
                    except FileNotFoundError:
                        st.warning(f"Missing file")
            
            if retrieved_contents:
                # Display answer in chat bubble
                combined_text = "\n".join(retrieved_contents)
                context = combined_text[:8000]

                prompt = f"""
                You are an expert in agricultural machine learning systems. Use the following context from research papers to answer the user's question. 
                Only use information from the context. Do not include paper names or authors. Do not fabricate content. Be precise and clear.

                Context:
                \"\"\"{context}\"\"\"

                Question:
                {query}

                Answer:
                """

                response = chat.generate_content(prompt)

                # Display response with markdown support
                ai_message(GoogleTranslator(source='en', target=lang_code).translate(response.text))

                # Enhanced image results section
                if show_images:
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 20px 0;">
                        <h3 style="color: #4CAF50;">🖼️ Relevant Visual Content</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ... [IMAGE RETRIEVAL CODE] ...
                    image_index = faiss.read_index(IMG_INDEX_PATH)
                    image_paths_all = np.load(IMG_META_PATH, allow_pickle=True)

                    clip_query_embedding = get_clip_text_embedding(query)

                    image_embeddings = image_index.reconstruct_n(0, image_index.ntotal)
                    similarities = cosine_similarity(clip_query_embedding, image_embeddings)[0]

                    image_scores = list(zip(image_paths_all, similarities))
                    image_scores.sort(key=lambda x: x[1], reverse=True)

                    top_images = [img for img in image_scores if img[1] > 0.30][:1]

                    if top_images:
                        for img_path, score in top_images:
                            # Resize image for display
                            img = Image.open(img_path)
                            img.thumbnail((150, 150))  # Resize image to small size
                            st.image(img,use_column_width=False)

                            # Show mapped paragraph if available
                            img_name = os.path.basename(img_path)
                            doc_key = os.path.basename(os.path.dirname(img_path))
                            img_text_key = f"{doc_key}/{img_name}"

                            # Load the image-to-text map from disk
                            if os.path.exists(IMAGE_TEXT_MAPPING_PATH):
                                img_text_map = np.load(IMAGE_TEXT_MAPPING_PATH, allow_pickle=True).item()
                                if img_text_key in img_text_map:
                                    st.markdown(f"**{t('Mapped Paragraph:', lang_code)}**\n\n{GoogleTranslator(source='en', target=lang_code).translate(img_text_map[img_text_key])}")
                                
                                #display_image(img_path)
                    else:
                        st.info("No highly relevant images found for this query.")
            else:
                ai_message("I couldn't find relevant information in the documents to answer that question.")
                            
        except Exception as e:
            ai_message(f"⚠️ Sorry, I encountered an error processing your request: {str(e)}")

# Helper function for image conversion
def img_to_base64(image):
    import io
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
