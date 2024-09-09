import pandas as pd
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

hf_api_key = "Your_HF_API_KEY"

# Load the CSV file
df = pd.read_csv("med_1.csv")

# Convert rows to a list of dictionaries or text
data_entries = df.to_dict(orient="records")

persist_directory = 'db'

embeddings = HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2',
                           huggingfacehub_api_token=hf_api_key)

texts = [f"Disease: {row['Disease']}, Symptoms: {row['Symptoms']}, Medications: {row['Medications']}" for row in data_entries]

# Generate embeddings for each text entry
vdb = Chroma.from_texts(texts, embeddings)

retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 1})

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    temperature=0.3,
    repetition_penalty=1.03,
)  # Define your Hugging Face model
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Process user query

question = "I have symptoms of vomitting and nausea can you prescribe any medication"

response = qa_chain.invoke({"query": question})

print(response)
