from flask import Flask, render_template, request, jsonify
from chatbot import chat, rebuild_index, search_docs
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/documents", methods=["GET"])
def get_documents():
    try:
        files = []
        if os.path.exists(DOCUMENTS_DIR):
            for f in os.listdir(DOCUMENTS_DIR):
                path = os.path.join(DOCUMENTS_DIR, f)
                if os.path.isfile(path) and f.lower().endswith(".pdf"):
                    files.append({
                        "name": f,
                        "size": os.path.getsize(path),
                        "type": "pdf"
                    })
        return jsonify({"documents": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_document():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        uploaded_files = request.files.getlist("file")
        if not uploaded_files or uploaded_files[0].filename == "":
            return jsonify({"error": "No files selected for upload"}), 400
            
        saved_files = []
        for file in uploaded_files:
            if file and file.filename.lower().endswith(".pdf"):
                filename = secure_filename(file.filename)
                dest_path = os.path.join(DOCUMENTS_DIR, filename)
                file.save(dest_path)
                saved_files.append(filename)
                
        if not saved_files:
            return jsonify({"error": "Only PDF files are supported"}), 400
            
        # Rebuild FAISS index
        rebuild_index()
        
        return jsonify({"message": f"Successfully uploaded {len(saved_files)} file(s).", "files": saved_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    try:
        filename = secure_filename(filename)
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        os.remove(file_path)
        
        # Rebuild index
        rebuild_index()
        
        return jsonify({"message": f"Successfully deleted {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    active_docs = data.get("active_docs", None)

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        start_time = time.time()
        result = chat(question, active_docs)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time (Sim Only): {elapsed_time:.2f} seconds")
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "elapsed_seconds": elapsed_time
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    active_docs = data.get("active_docs", None)

    try:
        max_results = int(data.get("max_results", 8))
    except (TypeError, ValueError):
        max_results = 8

    if not query:
        return jsonify({"error": "Empty search query"}), 400

    try:
        results = search_docs(query, active_docs=active_docs, max_results=max_results)
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002)
