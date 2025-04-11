import os
import time
import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import re

# Set Page Configuration
st.set_page_config(page_title="📸 AI Chatbot with Image Retrieval", page_icon="🤖", layout="wide")

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyD3wWkZBCq4ItfipexZ7NiUQLCAsw8IeCs"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure necessary folders exist
os.makedirs("extracted_texts", exist_ok=True)
os.makedirs("extracted_images", exist_ok=True)

# Load FAISS index and metadata
if os.path.exists("document_index.faiss") and os.path.exists("metadata.npy"):
    index = faiss.read_index("document_index.faiss")
    file_names = np.load("metadata.npy")
else:
    index, file_names = None, []

# Sidebar for Uploading Documents
st.sidebar.title("📂 Upload New Documents")
uploaded_files = st.sidebar.file_uploader("Drop PDFs here", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.success("Processing new documents...")
    new_files = []
    for uploaded_file in uploaded_files:
        text_file_path = os.path.join("extracted_texts", uploaded_file.name + ".txt")
        image_folder = os.path.join("extracted_images", uploaded_file.name)

        if not os.path.exists(text_file_path) or not os.path.exists(image_folder):
            os.makedirs(image_folder, exist_ok=True)
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for i, page in enumerate(pdf.pages):
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n\n"
                    for j, img in enumerate(page.images):
                        try:
                            image_data = img["stream"].get_data()
                            image_obj = Image.open(BytesIO(image_data))

                            # Convert CMYK/P mode images to RGB
                            if image_obj.mode in ["CMYK", "P"]:
                                image_obj = image_obj.convert("RGB")

                            # Get Image Size
                            width, height = image_obj.size

                            # **Filter Conditions**: Avoid small decorative images
                            if width < 100 or height < 100:  # Ignore very small images
                                continue
                            if width / height > 5 or height / width > 5:  # Ignore long banners
                                continue
                            if image_obj.mode == "L":  # Ignore grayscale images
                                continue

                            img_path = os.path.join(image_folder, f"image_{i}_{j}.png")
                            image_obj.save(img_path, format="PNG")
                        except (UnidentifiedImageError, Exception) as e:
                            print(f"Skipping invalid image: {e}")

            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            new_files.append(uploaded_file.name)

    if new_files:
        st.sidebar.success(f"New files extracted: {', '.join(new_files)}")
        st.sidebar.warning("Restart the app to index new documents.")
    else:
        st.sidebar.info("No new files detected. All uploaded files were already extracted.")

# Function to search documents
def search_documents(query, top_k=3, threshold=1.5):
    if index is None:
        return ["No documents indexed. Please upload files first."]
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] > threshold:
            continue
        file_name = file_names[idx]
        with open(f"extracted_texts/{file_name}", "r", encoding="utf-8") as f:
            content = f.read()
        results.append(content[:1000])
    return results if results else ["No relevant documents found."]

# Function to check if user requested images
def check_for_image_request(query):
    keywords = ["image", "photo", "figure", "diagram", "chart", "table", "illustration"]
    return any(word in query.lower() for word in keywords)

# Function to extract the number of images requested
def extract_image_count(query):
    match = re.search(r"(\d+)\s*(images|photos|pictures|figures)", query, re.IGNORECASE)
    return int(match.group(1)) if match else 2  # Default to 2 images if no number is mentioned

# Function to retrieve a specific number of images
def get_images(num_images=2):
    image_files = []
    for root, _, files in os.walk("extracted_images"):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                if os.path.exists(img_path):
                    image_files.append(img_path)

    return image_files[:num_images]  # Return only the requested number of images

# Function to generate chat response
def generate_response(query):
    print(f"🔍 Received query: {query}")  # Debugging
    if check_for_image_request(query):
        num_images = extract_image_count(query)  # Extract requested number of images
        images = get_images(num_images)
        if images:
            st.session_state["last_images"] = images
            return f"✅ Retrieved {len(images)} images successfully."
        return "❌ No images found in the uploaded documents."

    st.session_state["last_images"] = []  # Clear images when switching to text responses

    retrieved_docs = search_documents(query)
    if not retrieved_docs or retrieved_docs == ["No relevant documents found."]:
        return "❌ Sorry, I couldn't find any relevant information in the uploaded documents."

    context = "\n\n".join(retrieved_docs)
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    try:
        response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
        return response.text if response and hasattr(response, "text") else "⚠️ Error generating response."
    except Exception as e:
        return f"⚠️ AI model error: {str(e)}"

# Chat UI
st.title("📸 AI Chatbot with Image Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_images" not in st.session_state:
    st.session_state.last_images = []

query = st.chat_input("Ask me anything...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("AI is thinking..."):
        response = generate_response(query)
    
    if response:
        st.session_state.messages.append({"role": "ai", "content": response})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display retrieved images in a responsive grid layout
images = st.session_state.get("last_images", [])
if images:
    num_cols = 4  # Adjust the number of columns per row
    rows = [images[i : i + num_cols] for i in range(0, len(images), num_cols)]
    
    for row in rows:
        cols = st.columns(len(row))
        for col, img_path in zip(cols, row):
            col.image(img_path, caption="Retrieved Image", width=150)
