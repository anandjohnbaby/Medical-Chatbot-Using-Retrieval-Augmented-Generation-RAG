# Medical Q&A Chatbot

## Overview
This project implements a **Medical Q&A Chatbot** using **retrieval-augmented generation (RAG)** combined with **semantic search** and **large language model (LLM) response generation**. The chatbot retrieves relevant medical answers from a dataset and enhances them using **Gemini AI** for a more refined response.

## Features
- **Semantic Search with Sentence Embeddings**: Converts medical questions into vector embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`).
- **FAISS Index for Efficient Retrieval**: Stores and searches embeddings using **FAISS** (Facebook AI Similarity Search) to find the most relevant pre-existing medical question.
- **Retrieval-Augmented Generation (RAG)**: The retrieved answer is provided as context to **Gemini AI (LLM)** (`gemini-pro` model) for response generation.
- **Streamlit UI for User Interaction**: Provides an interactive chatbot interface using **Streamlit**.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed along with the required dependencies.

### Clone the Repository
```bash
git clone https://github.com/your-username/medical-qa-chatbot.git](https://github.com/anandjohnbaby/Medical-Chatbot-Using-Retrieval-Augmented-Generation-RAG.git
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset
The chatbot uses a medical Q&A dataset (`medquad.csv`). Ensure this dataset is available in the project directory.

## Running the Application
To start the chatbot, run:
```bash
streamlit run app.py
```

## How It Works
1. **Loading the Dataset**: Reads the `medquad.csv` file containing medical questions and answers.
2. **Embedding Generation**:
   - If precomputed embeddings exist (`question_embeddings.npy` and `faiss_index.bin`), they are loaded.
   - Otherwise, embeddings are computed and stored using **FAISS** for fast retrieval.
3. **User Query Processing**:
   - Converts the user’s question into an embedding.
   - Searches FAISS for the most relevant stored question.
   - Retrieves the corresponding answer.
4. **Response Generation**:
   - Sends the retrieved answer as context to **Gemini AI** for elaboration.
   - Displays the refined answer in the Streamlit interface.

## Technologies Used
- **Python**
- **FAISS** (Facebook AI Similarity Search)
- **SentenceTransformers** (`all-MiniLM-L6-v2`)
- **Gemini AI (LLM)** (`gemini-pro` model)
- **Streamlit** (Web UI)
- **NumPy, Pandas, OS** (Data handling)

## Future Enhancements
- Add support for more medical datasets.
- Implement multi-language support.
- Improve response generation using fine-tuned LLMs.

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues.

## Contact
For any questions, reach out via [anandjohnbabyv4@gmail.com] or open an issue on GitHub.
