import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
# from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


working_dir = os.path.dirname(os.path.abspath(__file__))

llm = Ollama(
    model = "llama3:instruct",
    temperature = 0
)

embeddings = HuggingFaceEmbeddings()

def get_answer(file_name, query):
    file_path = f"{working_dir}/{file_name}"

    #loading the document
    loader = PyPDFLoader(file_path)
    document = loader.load()

    #create text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(document)

    #vector embeddings from text chunks
    knowledge_base = FAISS.from_documents(chunked_documents, embeddings)

    # retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())
    response = qa_chain.invoke({"query": query})
    return response["result"]



    
