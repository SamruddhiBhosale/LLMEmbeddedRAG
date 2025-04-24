import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os
import pickle

st.set_page_config(page_title="📄 Document Q&A with Local LLM")
st.title("📄 Ask Questions About Your Document")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Set up paths for persistence
persist_dir = "faiss_index"
metadata_path = os.path.join(persist_dir, "doc_metadata.pkl")

if uploaded_file is not None:
    should_embed_new_file = True

    if os.path.exists(persist_dir) and os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            prev_file_name = pickle.load(f)

        if uploaded_file.name == prev_file_name:
            st.info(f"🔁 Reusing vector store from previously uploaded file: `{prev_file_name}`")
            should_embed_new_file = False
            db = FAISS.load_local(
                persist_dir,
                SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                allow_dangerous_deserialization=True
            )

    if should_embed_new_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Embed and store
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(persist_dir)

        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(uploaded_file.name, f)

        st.success("✅ Document embedded and vector store saved!")

        # Clean up temp file
        os.remove(temp_pdf_path)

    # Create retriever + LLM
    retriever = db.as_retriever()
    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question:")

    if query:
        result = qa.run(query)
        st.markdown("### 📌 Answer:")
        st.write(result)
