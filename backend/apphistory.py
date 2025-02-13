import os
import datetime
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__, template_folder="templates")
app.secret_key = "your_secret_key"  # Change this in production

# Ensure necessary directories exist
pdf_dir = "pdf"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    
# Folder where the vector store will persist
folder_path = "db"

# Global model configuration. Initially set to "llama3"
current_model = "llama3"
cached_llm = Ollama(model=current_model)

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=80, 
    length_function=len, 
    is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
    """
)

# ------------------------------
# User management and chat storage
# ------------------------------
# Simple in-memory user store.
# Format: { username: { "password": "plaintext-or-hashed", "chats": [ { "query": ..., "answer": ..., "timestamp": ... }, ... ] } }
users = {}

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    if username in users:
        return jsonify({"error": "Username already exists"}), 400
    users[username] = {"password": password, "chats": []}
    return jsonify({"message": "User registered successfully"})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    if username not in users or users[username]["password"] != password:
        return jsonify({"error": "Invalid credentials"}), 401
    session["username"] = username
    return jsonify({
        "message": "Logged in successfully", 
        "username": username, 
        "chats": users[username]["chats"]
    })

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    return jsonify({"message": "Logged out successfully"})

@app.route("/get_chats", methods=["GET"])
def get_chats():
    username = session.get("username")
    if username:
        return jsonify({"chats": users[username]["chats"]})
    else:
        return jsonify({"error": "User not logged in"}), 401

# ------------------------------
# PDF Management Endpoints
# ------------------------------

# List all uploaded PDF files
@app.route("/pdfs", methods=["GET"])
def list_pdfs():
    files = os.listdir(pdf_dir)
    # Optionally, filter for PDFs only:
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    return jsonify({"pdfs": pdf_files})

# Delete a PDF file by filename
@app.route("/pdf/<filename>", methods=["DELETE"])
def delete_pdf(filename):
    file_path = os.path.join(pdf_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"{filename} deleted successfully."})
    else:
        return jsonify({"error": "File not found."}), 404

# ------------------------------
# Main application endpoints
# ------------------------------

# Serve the HTML chat page
@app.route("/")
def index():
    return render_template("index.html")

# Return a list of available models (hard-coded for this example)
@app.route("/models", methods=["GET"])
def get_models():
    models = ["llama3", "gpt4", "llama2", "vicuna-7B"]
    return jsonify({"models": models})

# Set the current model by updating the global cached_llm
@app.route("/set_model", methods=["POST"])
def set_model():
    global cached_llm, current_model
    json_content = request.json
    new_model = json_content.get("model")
    if new_model:
        current_model = new_model
        cached_llm = Ollama(model=current_model)
        return jsonify({"status": "Model updated", "model": current_model})
    else:
        return jsonify({"status": "No model provided"}), 400

# A simple chat endpoint (general chat)
@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    
    # Save chat if the user is logged in
    username = session.get("username")
    if username:
        users[username]["chats"].append({
            "query": query,
            "answer": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    return jsonify({"answer": response})

# A retrieval-based PDF chat endpoint
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)

    sources = []
    for doc in result.get("context", []):
        sources.append({
            "source": doc.metadata.get("source", "unknown"), 
            "page_content": doc.page_content
        })
    
    # Save chat if the user is logged in
    username = session.get("username")
    if username:
        users[username]["chats"].append({
            "query": query,
            "answer": result["answer"],
            "timestamp": datetime.datetime.now().isoformat()
        })

    return jsonify({"answer": result["answer"], "sources": sources})

# PDF upload endpoint. Expects a file field named "file"
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join(pdf_dir, file_name)
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding, 
        persist_directory=folder_path
    )
    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return jsonify(response)

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
