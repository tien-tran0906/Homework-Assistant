import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

api = ''
st.title('🦜🔗 GPT ')


with st.sidebar:
    api_key = st.text_input('Input your OpenAI API Key')

if api_key:
    # Set APIkey for OpenAI Service
    os.environ['OPENAI_API_KEY'] = api_key

    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.1, verbose=True)
    embeddings = OpenAIEmbeddings()

    # Create and load PDF Loader
    loader = PyPDFLoader('Sommerville-Software-Engineering-10ed.pdf')
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, embeddings, collection_name='SWE')

    vectorstore_info = VectorStoreInfo(
        name="SWE",
        description="a SWE book",
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)


    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )


    prompt = st.text_input('Input your prompt here')

    # If the user hits enter
    if prompt:
        response = agent_executor.run(prompt)
        st.write(response)


        with st.expander('Document Similarity Search'):
            search = store.similarity_search_with_score(prompt)
            st.write(search[0][0].page_content)






