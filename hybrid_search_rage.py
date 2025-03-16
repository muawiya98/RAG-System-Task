from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain import hub
import numpy as np
import faiss
import uuid
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

loader = WebBaseLoader("https://www.wikipedia.org/")
data = loader.load()

embedding_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(data)

doc_ids = [str(uuid.uuid4()) for _ in documents]
document_texts = [doc.page_content for doc in documents]
document_embeddings = embedding_model.embed_documents(document_texts)
embedding_matrix = np.array(document_embeddings, dtype=np.float32)
embedding_dimension = embedding_matrix.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dimension)
faiss_index.add(embedding_matrix)

docstore = InMemoryDocstore({doc_ids[i]: documents[i] for i in range(len(documents))})
index_to_docstore_id = {i: doc_ids[i] for i in range(len(documents))}

vector_store = FAISS(
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_model.embed_query
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

llm = ChatOpenAI(model="gpt-4o")

prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/query", methods=["POST"])
def query_rag():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    try:
        response = qa_chain.invoke({"query": user_query})
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error in LangChain Classes: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
