from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_objectbox.vectorstores import ObjectBox
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain import hub
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
loader = WebBaseLoader("https://www.wikipedia.org/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = text_splitter.split_documents(data)

vector = ObjectBox.from_documents(documents, OpenAIEmbeddings(), embedding_dimensions=768)
# vector = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="./chroma_db")

llm = ChatOpenAI(model="gpt-4o")

prompt = hub.pull("rlm/rag-prompt")

# search_type="similarity", search_kwargs={"k": 1}
# search_type="mmr", search_kwargs={"k": 1, "fetch_k": 10}
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
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