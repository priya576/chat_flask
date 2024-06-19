from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, try to provide some correct answer on your knowledge, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGroq(api_key= os.getenv("GROQ_API_KEY"),model="llama3-8b-8192",temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

query = "I am having fever for 3-4 days.what disease it may be"

@app.route('/')
def home():
    return "Welcome to the Flask Backend!"

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        embeddings = download_hugging_face_embeddings()
        print("Doing it.....")

        new_db = FAISS.load_local("faiss_index_pk", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        print("almost done")
        chain = get_conversational_chain()
        print("Going Well")

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print(response["output_text"])
        return jsonify({"reply": response["output_text"]}), 200
    except Exception as e:
        return jsonify({"error": "An error occurred while processing your request"}), 500
    
