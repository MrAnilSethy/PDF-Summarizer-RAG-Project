#document loader
#text-splitter
#embeddings
#databse chroma-db
#reterivers
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

#document loader
loader = PyPDFLoader("deep-learning.pdf")
docs = loader.load()

#splitting chunking
splitter = RecursiveCharacterTextSplitter(    
    chunk_size=1000, chunk_overlap=200
)
chunks = splitter.split_documents(docs)

#embeddings 
embedding_model = MistralAIEmbeddings()

#chroma-db
vectorstores = Chroma.from_documents(
    documents=chunks,
    embedding = embedding_model,
    persist_directory="chroma_db"  
)

