from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI
from sqlite_memory import SQLiteMemory
import os
import requests
import json
from dotenv import load_dotenv

# Load configuration
with open('ds_config.json', 'r') as config_file:
    config = json.load(config_file)
load_dotenv()

class BearerAuth:
    def __init__(self):
        self.url = os.getenv("PING_FED_URL")
        self.client_id = os.getenv("KGW_CLIENT_ID")
        self.client_secret = os.getenv("KGW_CLIENT_SECRET")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }

        response = requests.post(self.url, headers=headers, data=data)

        if not response.ok:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        dict_of_response_text = json.loads(response.text)
        self.bearer_token = dict_of_response_text.get("access_token")

def create_chat_openai_client():
    auth = BearerAuth()
    client = AzureChatOpenAI(
        api_version=config['openai_api_version'],
        azure_endpoint=config["openai_url"],
        azure_ad_token=auth.bearer_token,
        azure_deployment=config['openai_model'],
    )
    return client

def create_openai_client_emb():
    auth = BearerAuth()
    client = AzureOpenAI(
        api_version=config['openai_api_version'],
        azure_endpoint=config["openai_url"],
        azure_ad_token=auth.bearer_token,
        azure_deployment=config['embedding_model'],
    )
    return client

class AzureEmbeddingFunction(Embeddings):
    def __init__(self, embedding_client, model_name):
        self.embedding_client = embedding_client
        self.model = model_name

    def embed_documents(self, texts):
        response = self.embedding_client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text):
        response = self.embedding_client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

def get_embedding_model():
    embedding_client = create_openai_client_emb()
    embedding_model = AzureEmbeddingFunction(embedding_client, config['embedding_model'])
    return embedding_model

def load_vectorstore():
    DB_FAISS_PATH = "FINAL_VECTOR_DB/"
    embedding_model = get_embedding_model()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db, embedding_model

def create_agent_with_memory(session_id):
    llm = create_chat_openai_client()
    db, embedding_model = load_vectorstore()
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        llm=llm
    )

    rag_tool = create_retriever_tool(
        retriever=retriever,
        name="pdf_knowledge_base",
        description="Answers questions using internal PDF documents"
    )

    memory = SQLiteMemory(session_id=session_id)

    agent = initialize_agent(
        tools=[rag_tool],
        llm=llm,
        memory=memory,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False
    )

    # Return all three components that main.py needs
    return agent, db, embedding_model