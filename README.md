
# RAG Model for Pharmacy and Healthcare

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
- **Pharmacy-related Dataset**: Contains medical and pharmacy-related documents for answering health queries.

## Stack Used

- **LLM**: [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- **Retrieval Chain**: LangChain's `RetrievalQA` for retrieving relevant documents and answering user queries.
- **Vector Store**: ChromaDB for storing and retrieving document vectors.
- **Embeddings**: Hugging Face's `'sentence-transformers/all-MiniLM-L6-v2'` for generating document embeddings.
- **Deployment**: The app is deployed using Streamlit for a simple and interactive user interface.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Saahil97/RetrievalAugmentedGeneration_QAChatbot.git
cd RetrievalAugmentedGeneration_QAChatbot
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

Alternatively, you can hardcode the API key in the code for local development.

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

- **Format**: The documents are stored as PDF files and indexed using ChromaDB.
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
