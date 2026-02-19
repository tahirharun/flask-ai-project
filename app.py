from flask import Flask, render_template, request, jsonify
from ai_module import get_ai_response, reset_chat

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# AI response endpoint
@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    ai_reply = get_ai_response(user_message)
    return jsonify({"response": ai_reply})

@app.route("/reset", methods=["POST"])
def reset():
    reset_chat()
    return jsonify({"status": "Chat reset!"})
if __name__ == "__main__":
    app.run(debug=True)
