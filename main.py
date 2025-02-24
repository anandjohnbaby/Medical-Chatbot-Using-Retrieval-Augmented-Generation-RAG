import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Load dataset
df = pd.read_csv("medquad.csv")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths for saved embeddings and FAISS index
EMBEDDINGS_PATH = "question_embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.bin"

# Check if embeddings and index exist
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Loading saved FAISS index and embeddings...")
    question_embeddings = np.load(EMBEDDINGS_PATH)

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("Computing embeddings and creating FAISS index...")

    # Convert questions to vector embeddings
    question_embeddings = np.array([embed_model.encode(q) for q in df["question"]])

    # Save embeddings
    np.save(EMBEDDINGS_PATH, question_embeddings)

    # Create FAISS index
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(question_embeddings)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

print("FAISS index ready!")

# Configure Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Function to search FAISS and generate response
def get_answer(user_question):
    user_embedding = np.array([embed_model.encode(user_question)])

    # Search FAISS for the closest question
    _, closest_idx = index.search(user_embedding, 1)
    best_match = df.iloc[closest_idx[0][0]]["answer"]

    # Generate response using Gemini
    prompt = f"User asked: {user_question}\n\nHere is a relevant answer from our dataset:\n{best_match}\n\nPlease improve and elaborate on this response."
    response = model.generate_content(prompt)

    return response.text


# # Example usage
# user_input = "What is Glaucoma?"
# response = get_answer(user_input)
# print(response)

# Streamlit UI
st.title("🩺 Medical Chatbot")
st.write("Ask me any medical question!")

# Maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a medical question...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    response = get_answer(user_input)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)