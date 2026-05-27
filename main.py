from langchain_classic.chains import RetrievalQA
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
#from langgraph.prebuilt import create_react_agent
import torch
import chromadb
import build_index



llm = ChatOpenAI(
    model_name = "google/gemma-3-4b",
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

#print(f"LLM: {llm}")
#print(f"Retriever: {retriever}")


qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

query = "Ile trwa kadencja sejmu?"
result = qa_chain.invoke({"query": query})

print("Odpowiedź:", result["result"])
print("\nŹródła:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")  # Pierwsze 100 znaków fragmentu

classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
qa_pipeline = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
#prompt = PromptTemplate.from_template("Odpowiedz na pytanie dotyczące kontekstu: {context}\nQuestion: {question}")


@tool
def sentiment_analysis_tool(text: str) -> str:
    """Przeprowadź analizę na podstawie podanego tekstu."""
    result_an = classifier(text)
    return str(result_an)

@tool
def document_qa_tool(question: str, context: str) -> str:
    """Odpowiedz na pytania dotyczące kontekstu danego dokumentu."""
    result_qa = qa_pipeline(question=question, context=context)
    return str(result_qa)

@tool
def search_documents(query: str) -> str:
    """ Przeszukuj artykuły tekstu Konstytucji
    Użyj tego gdy użytkownik pyta o Konstytucje.
    Zwaracaj fragmenty artykułów wraz ze źródłami
    """
    result_doc = qa_chain.invoke(query, n_results=3)
    sources = "\n".join([f"- {doc.page_content[:200]}..." for doc in result_doc["source_documents"]])
    return f"Odpowiedź z bazy: {result_doc['result']}\n\nŹródła fragmentów:\n{sources}"


tools = [search_documents, sentiment_analysis_tool, document_qa_tool]

sytem_prompt = """Jesteś pomocnym asystentem. Odpowiadaj zawsze po polsku.
Masz do dyspozycji narzędzia do przeszukiwania Konstytucji, analizy sentymentu oraz QA dokumentów.
Jeśli użytkownik pyta o Konstytucję, obowiązkowo użyj narzędzia `search_documents`.
Zawsze staraj się cytować źródła lub odnosić się do fragmentów uzyskanych z narzędzi.
Jeśli w bazach nie ma odpowiedzi na pytanie, odpowiedz szczerze: "Nie wiem"."""




agent = create_agent(model=llm, tools=tools, system_prompt=sytem_prompt)

ag_result = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Ile trwa kadencja sejmu w Polsce"}]
})
print(ag_result["messages"][-1].content)

