""" This is the working version of the multi-pdf-chat app using 
LangChain Classic and Groq LLM. It allows users to upload 
multiple PDF files, processes them to extract text, 
splits the text into manageable chunks, creates a vector store 
for efficient retrieval, and enables conversational querying of 
the content within the PDFs.
""" 
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Explicitly pull the API key from your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Reading pdf files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Splitting the text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

# Creating a vector store from the text chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(vector_store):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest question, "
                   "rephrase it into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the context below. "
                   "If the answer isn't in the context, say so.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = get_conversational_chain(db)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    response = chain.invoke({
        "input": user_question,
        "chat_history": st.session_state.chat_history
    })

    st.write("Reply:", response["answer"])
    st.session_state.chat_history.append(("human", user_question))
    st.session_state.chat_history.append(("ai", response["answer"]))

def main():
    st.set_page_config(page_title="Multi PDF Chat")
    st.header("Chat with Multiple PDFs 📚")

    user_question = st.text_input("Ask a question from the PDFs")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Done! Ask your questions.")

if __name__ == "__main__":
    main()