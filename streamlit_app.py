import pandas as pd
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain.prompts import PromptTemplate

hf_api_key = "Your_HF_API_KEY"

# Load the CSV file
df = pd.read_csv("med_2.csv")

# Convert rows to a list of dictionaries or text
data_entries = df.to_dict(orient="records")

persist_directory = 'db'

@st.cache_resource
def llm_pipeline():
    pipe = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    do_sample=False,
    repetition_penalty=1.03,
)
    return pipe

@st.cache_resource   
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2',
                           huggingfacehub_api_token=hf_api_key)

    texts = [f"Disease: {row['Disease']}, Symptoms: {row['Symptoms']}, Medications: {row['Medications']}" for row in data_entries]

# Generate embeddings for each text entry
    vdb = Chroma.from_texts(texts, embeddings)

    template = """
    <human>:
    Context: {context}

    Question: {question}
    
    If the question asked does not match with the Context provided or is not related to health, just say "This question is beyond my scope." If the question is related to health or medical information then use the context provided to answer the question.

    """
    prompt = PromptTemplate(input_variables=["context","question"],template=template)
    chain_type_kwargs = {"prompt":prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vdb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa.invoke(instruction)
    source_documents = generated_text['source_documents']

    # Remove duplicates based on page_content
    unique_documents = []
    seen_content = set()

    for doc in source_documents:
        # Normalize the page_content by stripping spaces and converting to lowercase
        normalized_content = doc.page_content.strip().lower()
        
        if normalized_content not in seen_content:
            unique_documents.append(doc)
            seen_content.add(normalized_content)

    return format_result(generated_text, unique_documents)

def format_result(output, unique_documents):
    query = output.get('query', 'N/A')
    result = output.get('result', 'N/A')
    #source_documents = output.get('source_documents', [])

    # Function to split the text at full stops and format it with new lines
    def split_text_at_full_stops(text):
        sentences = [sentence.strip() for sentence in text.split('.')]  # Split at full stops and trim spaces
        return '\n'.join(sentence for sentence in sentences if sentence)  # Join with new lines, removing empty parts

    # Structuring the output with the result split at full stops
    formatted_result = f"**Query**: {query}\n\n**Result**: \n{split_text_at_full_stops(result)}\n\n**Source Documents**:\n"
    
    for i, doc in enumerate(unique_documents):
        formatted_result += f"{i + 1}. {doc.page_content}\n"
    
    return formatted_result

def main():
    st.title("Search your doc")
    question = st.text_area("Please ask your Question")
    if st.button("Search"):
        st.info("Your Question:" + question)
        #st.info("Your source documents: ")
        source_documents = process_answer(question)

        st.markdown(f"```{source_documents}```")

if __name__ == '__main__':
    main()