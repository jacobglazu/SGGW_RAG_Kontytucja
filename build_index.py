from langchain.vectorstores import Chroma
import lmstudio as llm
from lmstudio import LMStudioClient
from openai import OpenAI
import load

split_doc = load.split_documents
#model = llm.embeddings_model("text-embedding-nomic-embed-text-v1.5")
# Initialize client
#client = LMStudioClient("http://localhost:1234/v1")
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=split_doc,
    embedding=client.embeddings.create,
    persist_directory="./chroma_db"
)