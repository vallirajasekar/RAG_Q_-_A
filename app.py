import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv 


load_dotenv()

#os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
print('Got the key')

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")





llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma-7b-it")
prompt=ChatPromptTemplate.from_template(
    """
    Answer the Question based on the Provided Context only 
    Please Provide the most accurate Response
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        st.session_state.loader=PyPDFDirectoryLoader("research_paper")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:1])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


# Streamlit interface
st.title("RAG DOCUMENT Q AND A")
user_prompt = st.text_input('Enter Your Query')

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain properly
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': user_prompt})

    st.write(response['answer'])

    with st.expander('Document Similarity Search'):
        for doc in response.get('context', []):  # Use .get() to avoid KeyError
            st.write(doc.page_content)  # Assuming 'page_content' is a property of the doc
            st.write('------------')
