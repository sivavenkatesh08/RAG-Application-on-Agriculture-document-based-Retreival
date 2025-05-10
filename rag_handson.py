import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = "Your-api-key"  # Replace with your API key

pdf_path = "path-of-your-pdf"  # Corrected path
loader = PyPDFLoader(pdf_path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding)

retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

print("\n✅ Gemini PDF QA Bot is ready! Type your question below (type 'exit' to quit).\n")

while True:
    question = input("❓ Ask: ")
    if question.lower() in ['exit', 'quit']:
        print("👋 Exiting.")
        break
    response = qa_chain.run(question)
    print("🧠 Answer:", response, "\n")
