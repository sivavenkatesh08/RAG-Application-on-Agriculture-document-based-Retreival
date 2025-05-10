import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Step 1: Set Google API Key
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Step 2: MCP-style Context Manager (basic)
class ContextWindowManager:
    def __init__(self, max_chunks=5):
        self.memory = []
        self.max_chunks = max_chunks

    def update(self, new_context):
        self.memory.append(new_context)
        if len(self.memory) > self.max_chunks:
            self.memory.pop(0)

    def get_context(self):
        return "\n---\n".join(self.memory)

# Step 3: Load Documents
loader = TextLoader("your-file-path")  # Replace with your path
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 4: Create vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Step 5: Create LLM with Retriever
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 6: Start MCP-based loop
context_manager = ContextWindowManager()

print("\n🔄 Model Context Protocol (MCP) Assistant Ready. Type 'exit' to quit.\n")

while True:
    user_input = input("🗣️ You: ")
    if user_input.lower() == "exit":
        break

    # Get retrieved context
    response = qa_chain.run(user_input)
    context_manager.update(f"Q: {user_input}\nA: {response}")

    # Send full context to model (simulate MCP memory)
    full_context = context_manager.get_context()
    print(f"🤖 Gemini (MCP): {response}\n")
