#from langchain_chroma import Chroma
#import lmstudio as llm
#from lmstudio import LMStudioClient
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
import load

def main(embeddings):
    split_documents = load.main()
    print(split_documents)
    texts = [doc.page_content for doc in split_documents]
    #separator = "\n"
    #doc_str = separator.join([doc.page_content for doc in split_documents])
#model = llm.embeddings_model("text-embedding-nomic-embed-text-v1.5")
# Initialize client
#client = LMStudioClient("http://localhost:1234/v1")
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=60)

    #embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5", api_key="lm-studio")
    #embeddings = OpenAIEmbeddings(model="text-embedding-nomic-embed-text-v1.5",base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=60)
    vectorstore = None
    batch_size = 500
    for start in range(0, len(split_documents), batch_size):
        batch = split_documents[start: start + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vectorstore.add_documents(batch)


    #print(vectorstore)
    return vectorstore
if __name__ == "__main__":
    main()