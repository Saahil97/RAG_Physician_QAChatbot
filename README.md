# Physician Chatbot using RAG Model for Pharmacy and Healthcare

This repository contains a Retrieval-Augmented Generation (RAG) model designed to answer questions related to diseases, health issues, and medications. The model uses the **Zephyr-7B** LLM combined with document retrieval from a pharmacy-related dataset. It provides responses with medication recommendations or useful steps based on the provided question.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Stack Used](#stack-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Deployment](#deployment)
- [License](#license)

## Overview
This project leverages a **Retrieval-Augmented Generation (RAG)** approach to provide relevant answers to health-related queries. When the user asks about a disease, medication, or health issue, the system retrieves relevant documents from a custom pharmacy-related dataset and uses a large language model to generate responses based on the retrieved context.

## Features
- **Contextual Retrieval**: Uses a vector-based retrieval system to find relevant documents from the dataset.
- **LLM Response Generation**: The Zephyr-7B model generates responses using retrieved documents as context.
- **Streamlit UI**: A simple web interface for querying the model and getting responses.
- **ChromaDB for Document Storage**: Efficient and scalable document retrieval.
- **Medical Information Retrieval**: The chatbot provides information based on symptoms, diseases, and medications stored in a medical dataset.
- **Conversational Memory**: The chatbot remembers prior interactions with the user, making the conversation more natural and personalized.
- **Follow-Up Questions**: The chatbot suggests relevant follow-up questions to guide users through a diagnostic process, just like a real physician.
- **Non-Medical Query Filtering**: The chatbot is designed to only respond to health-related queries. Non-medical queries will receive a response saying "This question is beyond my scope."
- **Empathetic Responses**: The chatbot provides empathetic and supportive responses, ensuring users feel understood and guided appropriately.

## Stack Used

- **LLM**: [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- **Streamlit**: Web application framework for the frontend.
- **LangChain**: For chaining together models and data retrieval.
- **HuggingFace**: Used for model-based queries with the HuggingFace API.
- **Chroma**: Vector store for storing and retrieving document embeddings.
- **Pandas**: For handling and processing medical dataset CSV files.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Saahil97/RAG_Physician_QAChatbot.git
cd RAG_Physician_QAChatbot
```

### 2. Install Dependencies

Make sure you have Python 3.9 or later installed. Install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` includes necessary libraries such as `torch`, `transformers`, `langchain`, `chromadb`, `sentence-transformers`, `streamlit`, and others.

### 3. Set up ChromaDB

Ensure the ChromaDB vector store is set up correctly. You can modify the `persist_directory` in the code to point to your document database.

### 4. Hugging Face API Key

Get your API key from Hugging Face and set it as an environment variable:

```bash
export HF_API_KEY="your_huggingface_api_key"
```

## Usage

1. **Run the Streamlit App**

   Start the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

2. **Query the Model**

   You can ask questions related to diseases, health issues, or medications, and the model will retrieve relevant information from the dataset to provide answers. For example:

   - "What is diabetes, and how can it be treated?"
   - "What are the medications for high blood pressure?"

3. **View the Results**

   The app will display the generated answer along with the relevant documents used as the context.

## Dataset

The dataset used is a collection of pharmacy and healthcare-related documents. It includes information about diseases, treatments, medications, and medical conditions.

- **Format**: The documents are stored in .csv format, indexed using ChromaDB.
- **Embedding**: The dataset is embedded using Hugging Face’s `'sentence-transformers/all-MiniLM-L6-v2'`.

You can modify the dataset by adding or removing documents in the `db` directory and re-indexing them using ChromaDB.

## Deployment

The model is deployed using **Streamlit** for an interactive web-based user interface.

To deploy:

1. Make sure the repository and all dependencies are set up.
2. Run the following command to start the app locally:
   ```bash
   streamlit run app.py
   ```

You can also deploy this app on cloud services like **Heroku**, **AWS**, or **Streamlit Cloud**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Example Queries

- **Input**: "What are the symptoms of flu?"
- **Output**: 
  - **Generated Answer**: "The flu is typically characterized by fever, chills, muscle aches, cough, congestion, runny nose, headaches, and fatigue."
  - **Source Documents**: Relevant documents retrieved from the dataset. (Doc1, Doc2, etc;)

---
