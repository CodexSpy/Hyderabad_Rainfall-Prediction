from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json


with open("my_db.json", "r") as f:
    documents = json.load(f)

load_dotenv()
cohere_key = st.secrets["CO_API_KEY"]


embeddings = CohereEmbeddings(
    cohere_api_key=cohere_key,
    user_agent="rainfall-app/1.0"
)

docs = [
    Document(page_content=item["text"], metadata={"term": item["term"]})
    for item in documents
]


db = FAISS.from_documents(docs, embeddings)

terms = [item["term"] for item in documents]

llm = ChatCohere(
    model="command-r-plus",
    cohere_api_key=cohere_key
)

template = """
You are explaining a Data Science concept to a layperson.
Here is the context:
{context}

Explain this in simple terms with an example if possible.
"""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

__all__ = ["db", "terms", "chain"]
