from dotenv import load_dotenv
import os
import streamlit as st
import json

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import cohere

# Load .env and secrets
load_dotenv()
cohere_key = st.secrets.get("CO_API_KEY") or os.getenv("CO_API_KEY")

# Use official Cohere Python SDK for chat
co = cohere.Client(cohere_key)

# Load docs
with open("my_db.json", "r") as f:
    documents = json.load(f)

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

# The template is still useful
template = """
You are explaining a Data Science concept to a layperson.
Here is the context:
{context}

Explain this in simple terms with an example if possible.
"""
prompt = PromptTemplate.from_template(template)

# The wrapper for your chain → use official Cohere SDK directly
def run_chain(context: str):
    user_prompt = prompt.format(context=context)
    response = co.chat(
        model="command-r-plus",
        message=user_prompt
    )
    return response.text

__all__ = ["db", "terms", "run_chain"]
