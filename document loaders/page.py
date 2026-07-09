from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model = ChatGroq(model="openai/gpt-oss-safeguard-20b")
url = "https://www.apple.com/apple-vision-pro/"
data = WebBaseLoader(url)
docs = data.load()
template = ChatPromptTemplate([("system","You acta like a Apple seales man and detasils all the things"),("human","{data}")])

prompt = template.format_messages(data = docs)
response = model.invoke(prompt)
print(response.content)


