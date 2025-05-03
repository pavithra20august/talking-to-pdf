import streamlit as st
from pdf_loader import extract_text_from_pdf
from chatbot import get_vectorstore, get_qa_chain
import tempfile

st.set_page_config(page_title="PDF Q&A Chatbot")
st.title("RAG Open-Source Edition")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        text = extract_text_from_pdf(tmp.name)

    st.success("PDF Loaded. Ready to chat!")
    
    if 'qa_chain' not in st.session_state:
        vectorstore = get_vectorstore(text)
        st.session_state.qa_chain = get_qa_chain(vectorstore)

    query = st.text_input("Ask a question about your PDF:")
    if query:
        answer = st.session_state.qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")