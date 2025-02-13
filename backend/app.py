from flask import Flask, request, render_template, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__, template_folder="templates")
app.secret_key = "your_secret_key"  # Make sure to change this for production!

# -------------------------------
# Database Configuration & Models
# -------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define a User model for registration (and later, chat storage if needed)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # You could add a chat_history field later (e.g., as a JSON column)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create the tables if they do not exist
with app.app_context():
    db.create_all()

# -------------------------------
# Registration Endpoint
# -------------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    # Check if the user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400

    # Create a new user and store the hashed password
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

# -------------------------------
# Existing App Code (Your Endpoints)
# -------------------------------

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

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = User.query.filter_by(username=username).first()
    if user is None or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Optionally, set a session variable to keep the user logged in
    session["username"] = username

    return jsonify({"message": "Login successful", "username": username})




# A simple chat endpoint (general chat)
@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
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

    return jsonify({"answer": result["answer"], "sources": sources})

# PDF upload endpoint. Expects a file field named "file"
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
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
