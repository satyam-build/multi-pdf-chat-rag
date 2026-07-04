import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
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

# Creating a vector store from the test chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")   # persist to disk

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "answer is not available in the context."
Do not guess.

Context:\n{context}\n
Question: {question}
Answer:
""",
)


def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
    )

    def chain(question):
        docs = vector_store.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    chain = get_conversational_chain(db)
    response = chain(user_question)
    st.write("Reply:", response)

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