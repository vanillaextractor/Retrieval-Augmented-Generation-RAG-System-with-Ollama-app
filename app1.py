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

# --- Page Configuration ---
st.set_page_config(
    page_title="Ask Questions About Your PDF",
    page_icon="📄",
    layout="wide"
)

# --- Prompt Template ---
# This template guides the LLM to answer based on the provided context.
prompt_template = PromptTemplate.from_template(
    """
    Answer the question based only on the context provided below.
    Your answer should be thorough, clear, and directly extracted from the text.
    If the information to answer the question is not in the context, respond with "BHAI mujhe nahi pata ye sab, meine nahi padha hai ", but the information is not available in the provided document."

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
    st.title("📄 Ask Questions About Your PDF")
    st.markdown("Upload a PDF file and ask any question about its content. The app will use a local LLM to find the answer for you.")

    # --- Sidebar for PDF Upload ---
    with st.sidebar:
        st.header("1. Upload Your PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        st.info("Make sure your local Ollama server is running with the `llama3` and `nomic-embed-text` models available.")

    # Initialize session state variables to hold data across reruns
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

    # --- PDF Processing ---
    if uploaded_file:
        # Process the PDF only if it's a new file
        if st.session_state.pdf_name != uploaded_file.name:
            st.session_state.pdf_name = uploaded_file.name
            with st.spinner("Processing PDF... This may take a moment."):
                try:
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # 1. Load and split the PDF document
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load_and_split()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = splitter.split_documents(pages)

                    # 2. Create embeddings and store them in a Chroma vector store
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vectorstore = Chroma.from_documents(
                        documents=split_docs,
                        embedding=embeddings,
                        collection_name=f"pdf-{uploaded_file.name}" # Unique collection for each PDF
                    )

                    # 3. Create the retriever
                    st.session_state.retriever = vectorstore.as_retriever()

                    # 4. Initialize the LLM and the processing chain
                    llm = Ollama(model="llama3.2")
                    parser = StrOutputParser()
                    st.session_state.chain = prompt_template | llm | parser

                    st.success(f"✅ PDF '{uploaded_file.name}' processed! You can now ask questions.")

                except Exception as e:
                    st.error(f"An error occurred while processing the PDF: {e}")
                finally:
                    # Clean up the temporary file
                    if 'pdf_path' in locals() and os.path.exists(pdf_path):
                        os.remove(pdf_path)

    # --- Question and Answer Section ---
    st.header("2. Ask a Question")

    if st.session_state.retriever:
        question = st.text_input(
            "Ask a question about the content of the PDF:",
            placeholder="e.g., What is the main conclusion of the document?",
            key="question_input"
        )

        if question:
            with st.spinner("Finding the answer..."):
                try:
                    # Retrieve relevant documents
                    docs = st.session_state.retriever.get_relevant_documents(question)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Invoke the chain to get the response
                    response = st.session_state.chain.invoke({
                        "context": context,
                        "question": question
                    })

                    # Display the answer
                    st.markdown("### 🤖 Answer")
                    st.write(response)

                    # Display the sources (context) used for the answer
                    with st.expander("Show Sources"):
                        st.info("The answer was generated based on the following text snippets from the PDF:")
                        st.markdown(f"```text\n{context}\n```")
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
    else:
        st.warning("Please upload a PDF file first to enable the question and answer section.")


if __name__ == "__main__":
    main()
