from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()
model = ChatMistralAI(model="mistral-small-latest")
from langchain_core.documents import Document

docs = [
    Document(
        page_content="""
        Retrieval-Augmented Generation (RAG) is an AI technique that combines
        a Large Language Model with external knowledge sources. It retrieves
        relevant information from documents before generating an answer,
        improving accuracy and reducing hallucinations.
        """,
        metadata={"source": "rag_notes.txt", "topic": "RAG"}
    ),

    Document(
        page_content="""
        LangChain is an open-source framework for building applications powered
        by Large Language Models. It provides components for prompts,
        document loaders, vector stores, retrievers, memory, and agents.
        """,
        metadata={"source": "langchain_notes.txt", "topic": "LangChain"}
    ),

    Document(
        page_content="""
        Vector databases such as FAISS and ChromaDB store embeddings of text.
        They perform similarity search to retrieve the most relevant document
        chunks for a user's query in RAG applications.
        """,
        metadata={"source": "vector_db_notes.txt", "topic": "Vector Database"}
    )
]

embedding_model = MistralAIEmbeddings()

vectorestore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorestore.similarity_search("What is RAG ?",2)
for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever = vectorestore.as_retriever()
docs = retriever.invoke()