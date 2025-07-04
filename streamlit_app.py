import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import bs4

import streamlit as st

# Load environment variables
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

# # Load data
# class_label = "beebom-single-content-container"
# web_url = "https://beebom.com/valorant-characters-agents-abilities"

# loader = WebBaseLoader(web_paths=[web_url],
#                 bs_kwargs={
#                     "parse_only": bs4.SoupStrainer(class_=class_label),
#                 })
# docs = loader.load()

# text_loader = TextLoader("data/lore.txt")
# text_docs = text_loader.load()

# docs.extend(text_docs)

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
# )

# splitted_docs = text_splitter.split_documents(docs)

# Generate embeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    multi_process=True,
    model_kwargs={
        "device": "cuda",
    },
    encode_kwargs={
        "normalize_embeddings": True,
    }
)

# Load vector database

vectorstore_db = FAISS.load_local("valorant_agents_db", embedder, allow_dangerous_deserialization=True)

# Create chain

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Use the following pieces of context wrapped within <context>..</context> to answer the question. If you can't answer the question based on the given context, just say that you don't know. Do not try to make up an answer.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

llm = ChatGroq(
    model_name="gemma2-9b-it",
    temperature=0.1
)

retriever = vectorstore_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
    }
)


context_info = RunnableParallel({
    "context": retriever,
    "input": RunnablePassthrough()
})

chain = context_info | prompt | llm | StrOutputParser()

st.title("Valorant Knowledge Base")

query = st.text_input("Enter your query regarding the game Valorant:")
if query:
    response = chain.invoke(query)
    st.write("Response:", response)
