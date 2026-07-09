import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Any PDF")

st.write("Upload a PDF and ask questions.")


embedding_model = MistralAIEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-2603"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant.

Answer ONLY using the given context.

If the answer is not found in the document say:

"I could not find the answer in the document."

Keep answers detailed and easy to understand.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)


uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.sidebar.button("Process PDF"):

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        with st.spinner("Reading PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())

                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            st.session_state.retriever = retriever
            st.session_state.docs = docs

        st.success("PDF processed successfully!")

if "docs" in st.session_state:

    if st.button("📄 Summarize PDF"):

        with st.spinner("Generating summary..."):

            text = "\n".join(
                [d.page_content for d in st.session_state.docs]
            )

            summary_prompt = f"""
Summarize the following PDF.

Make sections:

Overview

Important Topics

Key Points

Conclusion

{text[:25000]}
"""

            response = llm.invoke(summary_prompt)

            st.subheader("Summary")

            st.write(response.content)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about the PDF...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        if "retriever" not in st.session_state:

            st.write("Please upload and process a PDF first.")

        else:

            docs = st.session_state.retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(final_prompt)

            st.markdown(response.content)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )