import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from streamlit_chat import message
import docx

# Load geminiAI API environment variables from .env file
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        doc_reader = docx.Document(doc)
        for paragraph in doc_reader.paragraphs:
            text += paragraph.text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some documents and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    context = "\n".join([doc.page_content for doc in docs])
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def main():
    st.set_page_config(page_title="Document Assist Tool", layout="wide")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        word_docs = st.file_uploader("Upload your Word Files", accept_multiple_files=True, type=["docx", "doc"])
        if st.button("Process"):
            if pdf_docs or word_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs) + get_word_text(word_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one document.")

    st.title("Document Assist tool")
    st.write("Here to assist you with your documents")

    st.text_area("Extracted Text", get_pdf_text(pdf_docs) + get_word_text(word_docs) if pdf_docs or word_docs else "No Document uploaded", height=300)

    st.sidebar.button('Clear Chat', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload some Documents and ask me a question"}]

    for i, message_entry in enumerate(st.session_state.messages):
        if message_entry["role"] == "user":
            message(message_entry["content"], is_user=True, key=f"user_{i}")
        else:
            message(message_entry["content"], is_user=False, key=f"assistant_{i}")

    def on_input_change():
        user_question = st.session_state.user_input
        st.session_state.messages.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.user_input = ""  # Clear the input field after submitting

    st.text_input("Your question:", on_change=on_input_change, key="user_input")

if __name__ == "__main__":
    main()