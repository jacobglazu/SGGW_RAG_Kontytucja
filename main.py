from langchain.chains import RetrievalQA
from openai import OpenAi
import build_index



llm = OpenAi(
    model_name = "local-model",
    base_url = "http://localhost:1234/v1",
    api_key = "lm-studio",
    temperature = 0.5

)
db = build_index.vectorstore

retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_deocuments = True
)