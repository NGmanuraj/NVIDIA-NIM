import streamlit as st
import os
import fitz  # PyMuPDF
from io import BytesIO
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Define a Document class with metadata
class Document:
    def __init__(self, content, page_number=0, metadata=None):
        self.page_content = content
        self.page_number = page_number
        self.metadata = metadata if metadata is not None else {}

def load_pdf_from_bytes(pdf_bytes):
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    documents = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        documents.append(Document(text, page_num))
    return documents

def vector_embedding(documents):
    st.session_state.embeddings = NVIDIAEmbeddings()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents[:30])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

prompt1 = st.text_input("Enter Your Question From Documents")

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Load and process each uploaded file
    all_documents = []
    for uploaded_file in uploaded_files:
        pdf_bytes = BytesIO(uploaded_file.read())
        documents = load_pdf_from_bytes(pdf_bytes)
        all_documents.extend(documents)

    # Perform vector embedding with the loaded documents
    if st.button("Documents Embedding"):
        vector_embedding(all_documents)
        st.write("Vector Store DB Is Ready")

    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time :", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
