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
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Ask Questions About Your PDFs",
    page_icon="📚",
    layout="wide"
)

# --- Prompt Template ---
# This template guides the LLM to answer based on the provided context.
prompt_template = PromptTemplate.from_template(
    """
    Answer the question based only on the context provided below from multiple documents.
    Your answer should be thorough, clear, and synthesized from the information across the documents.
    If the information to answer the question is not in the context, respond with "Mujhe nahi pata meine nahi padha ye sab, padhke bata dunga"
    When you answer, mention which document the information is coming from if possible.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# --- Main Application ---
def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title("📚 Ask Questions About Your Documents")
    st.markdown("Upload one or more PDF files, and then ask any question about their combined content.")

    # --- Sidebar for PDF Upload and Controls ---
    with st.sidebar:
        st.header("1. Upload Your PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        st.info("Make sure your local Ollama server is running with the `llama3.2` and `nomic-embed-text` models available.")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


    # Initialize session state variables
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- PDF Processing ---
    if uploaded_files:
        uploaded_file_names = sorted([file.name for file in uploaded_files])

        if uploaded_file_names != st.session_state.processed_files:
            st.session_state.processed_files = uploaded_file_names
            
            with st.spinner("Processing PDFs... This may take a moment."):
                all_split_docs = []
                temp_files = []
                try:
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            pdf_path = tmp_file.name
                            temp_files.append(pdf_path)

                        loader = PyPDFLoader(pdf_path)
                        pages = loader.load()
                        for page in pages:
                            page.metadata['source'] = uploaded_file.name

                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        split_docs = splitter.split_documents(pages)
                        all_split_docs.extend(split_docs)

                    collection_name = f"collection-{int(time.time())}"
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vectorstore = Chroma.from_documents(
                        documents=all_split_docs,
                        embedding=embeddings,
                        collection_name=collection_name
                    )
                    st.session_state.retriever = vectorstore.as_retriever()
                    llm = Ollama(model="llama3.2")
                    parser = StrOutputParser()
                    st.session_state.chain = prompt_template | llm | parser
                    st.success(f"✅ PDFs processed: {', '.join(uploaded_file_names)}")

                except Exception as e:
                    st.error(f"An error occurred while processing the PDFs: {e}")
                finally:
                    for path in temp_files:
                        if os.path.exists(path):
                            os.remove(path)

    # --- Chat History Display ---
    st.header("2. Chat with Your Documents")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Question Input and Answering ---
    if st.session_state.retriever:
        if question := st.chat_input("Ask a question about your documents..."):
            # Add user message to history and display it
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Finding the answer..."):
                    try:
                        # Retrieve relevant documents
                        docs = st.session_state.retriever.get_relevant_documents(question)
                        
                        # Format the context
                        context_parts = []
                        for doc in docs:
                            source_info = doc.metadata.get('source', 'Unknown Document')
                            page_info = doc.metadata.get('page', 'N/A')
                            context_parts.append(f"Source: {source_info} (Page: {page_info})\nContent: {doc.page_content}")
                        context = "\n\n---\n\n".join(context_parts)

                        # Invoke the chain
                        response = st.session_state.chain.invoke({
                            "context": context,
                            "question": question
                        })

                        st.write(response)

                        # Display sources in an expander
                        with st.expander("Show Sources"):
                            st.info("The answer was generated based on the following text snippets:")
                            st.markdown(f"```text\n{context}\n```")
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    else:
        st.warning("Please upload one or more PDF files to begin the chat.")

if __name__ == "__main__":
    main()
