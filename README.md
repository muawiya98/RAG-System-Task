
# RAG System with Flask and Node.js

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain** for document processing, **Flask API** for the Python backend, and **Node.js API** as a proxy to handle queries. It uses **Wikipedia articles** for document retrieval and **OpenAI** embeddings for vector search. 

## Project Structure

1. **Flask API (rag_pipeline.py)**: Handles document loading, vector embedding, and querying the **LangChain** system.
2. **Node.js API (server.js)**: Acts as a proxy that forwards queries from the client to the Flask API and returns the result.
3. **Dependencies**: Required packages are listed below for both Python and Node.js.

## Requirements

### Python Dependencies:
- `langchain`
- `langchain-openai`
- `langchain-core`
- `langchain-text-splitters`
- `langchain-community`
- `langchain-objectbox`
- `flask`
- `openai`

You can install them using pip:
```
pip install langchain langchain-openai langchain-core langchain-text-splitters langchain-community langchain-objectbox flask openai
```

### Node.js Dependencies:
- `express`
- `axios`
- `cors`
- `body-parser`

You can install them using npm:
```
npm install express axios cors body-parser
```

## Setup Instructions

1. **Start the Flask API (Python Backend):**

   In the Python environment, run:
   ```bash
   python rag_pipeline.py
   ```
   This will start the Flask API on `http://127.0.0.1:5000`.

2. **Start the Node.js API (Frontend Proxy):**

   In the Node.js directory, run:
   ```bash
   node server.js
   ```
   This will start the Node.js API on `http://127.0.0.1:3000`.

3. **Query the Node.js API:**

   You can use tools like **Postman** or **cURL** to send POST requests to the **Node.js API** with a JSON payload:
   ```json
   { "query": "you question" }
   ```

   The Node.js server will forward the query to the Flask API, which will process it and return a response.

## Folder Structure

```
/project-root
  ├── flask_rag.py
  └── server.js
  └── package.json
```
