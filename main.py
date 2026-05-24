from langchain_classic.chains import RetrievalQA
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import chromadb
import build_index



llm = ChatOpenAI(
    model_name = "text-embedding-nomic-embed-text-v1.5",
    base_url = "http://localhost:1234/v1",
    api_key = "lm-studio",
    temperature = 0.5

)
#embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

#db = build_index.main(embeddings=embeddings)
db = Chroma(persist_directory="./chroma_db")
#db = chromadb.PersistentClient(path="./chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 3})

print(f"LLM: {llm}")
print(f"Retriever: {retriever}")


qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

query = "Czym jest Rzeczpospolita Polska według Konstytucji?"
result = qa_chain.invoke({"query": query})

print("Odpowiedź:", result["result"])
print("\nŹródła:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")  # Pierwsze 100 znaków fragmentu
