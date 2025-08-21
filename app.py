import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile

# Prompt template
prompt = PromptTemplate.from_template("""
Answer the question based on the context below.
If you can't answer the question based on the context, respond with "mujhe nahi pata".

Context:
{context}

Question:
{question}

Answer:
""")

# Initialize LLM
llm = Ollama(model="llama3")
parser = StrOutputParser()
chain = prompt | llm | parser

st.title("📄 Ask Questions About Your PDF")

# 1. Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # 2. Load and split PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(pages)

    # 3. Create embeddings and store in Chroma
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="uploaded_pdf"
    )
    retriever = vectorstore.as_retriever()

    st.success("✅ PDF processed! Now you can ask questions.")

    # 4. Question input
    question = st.text_input("Ask a question about the PDF:")

    if question:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = chain.invoke({"context": context, "question": question})
        st.markdown("### 🤖 Answer:")
        st.write(response)
