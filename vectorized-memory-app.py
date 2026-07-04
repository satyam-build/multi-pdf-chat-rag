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
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Check your .env file.")
    st.stop()

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# ── PDF ingestion ────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=get_embeddings())
    vector_store.save_local("faiss_index")

# ── Vector memory (replaces ConversationBufferMemory) ────────────
def init_memory_store():
    """Create an in-memory FAISS store for chat history embeddings."""
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = None   # empty until first exchange
    if "raw_history" not in st.session_state:
        st.session_state.raw_history = []      # keeps (role, text) pairs

def add_to_memory(human_msg: str, ai_msg: str):
    """Embed a Q&A turn and upsert it into the memory vector store."""
    turn_text = f"User: {human_msg}\nAssistant: {ai_msg}"
    embeddings = get_embeddings()
    if st.session_state.memory_store is None:
        st.session_state.memory_store = FAISS.from_texts(
            [turn_text], embedding=embeddings
        )
    else:
        st.session_state.memory_store.add_texts([turn_text])

def get_relevant_history(current_question: str, k: int = 3):
    """
    Retrieve the k most relevant past turns for the current question.
    Returns a list of LangChain message objects ready for the prompt.
    """
    if st.session_state.memory_store is None:
        return []

    docs = st.session_state.memory_store.similarity_search(
        current_question, k=k
    )
    messages = []
    for doc in docs:
        # Each doc is "User: ...\nAssistant: ..." — split and rebuild as messages
        lines = doc.page_content.split("\nAssistant: ", 1)
        if len(lines) == 2:
            human_text = lines[0].replace("User: ", "", 1)
            ai_text = lines[1]
            messages.append(HumanMessage(content=human_text))
            messages.append(AIMessage(content=ai_text))
    return messages

# ── Conversational chain ─────────────────────────────────────────
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

# ── Main query handler ───────────────────────────────────────────
def user_input(user_question):
    embeddings = get_embeddings()
    db = FAISS.load_local(
        "faiss_index", embeddings,
        allow_dangerous_deserialization=True
    )
    chain = get_conversational_chain(db)
    init_memory_store()

    # Only send the RELEVANT past turns, not the full history
    relevant_history = get_relevant_history(user_question, k=3)

    response = chain.invoke({
        "input": user_question,
        "chat_history": relevant_history      # ← max 3 relevant turns, not all
    })

    answer = response["answer"]
    st.write("Reply:", answer)

    # Store this turn in vector memory for future retrieval
    add_to_memory(user_question, answer)

# ── Streamlit UI ─────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Multi PDF Chat")
    st.header("Chat with Multiple PDFs 📚")

    init_memory_store()

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

        if st.button("Clear Memory"):
            st.session_state.memory_store = None
            st.session_state.raw_history = []
            st.success("Memory cleared.")

if __name__ == "__main__":
    main()