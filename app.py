from flask import Flask, render_template, request, jsonify
from chatbot import chat
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        start_time = time.time()
        answer = chat(question)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time (Sim Only): {elapsed_time:.2f} seconds")
        return jsonify({"answer": answer, "elapsed_seconds": elapsed_time})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002)
