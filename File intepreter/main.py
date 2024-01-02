import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
import os
import openai
from PIL import Image

image = Image.open('Unknown.png')

st.image(image)

# Chat title
st.header("ChatGPT file interpreter")
st.write("*Upload your own files and ask questions to ChatGPT*")
st.write('**File types supported: PDF/DOCX/TXT/CSV**')

# Load version history from the text file
def how_to():
    with open("how_to.txt", "r") as file:
        return file.read() 
with st.sidebar.expander("**How To**", expanded=False):
        st.write(how_to())



with st.sidebar:
    # Input for OpenAI API Key
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Check if OpenAI API Key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Set OPENAI_API_KEY as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChatOpenAI model
llm = ChatOpenAI(temperature=2, model_name="gpt-3.5-turbo-16k", streaming=True)


# Sidebar section for uploading files and providing a YouTube URL
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)

    st.info("Refresh the browser if you decide to upload more files to reset the session")

# Check if files are uploaded or YouTube URL is provided
if uploaded_files:
    # Print the number of files uploaded or YouTube URL provided to the console
    st.write(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded files
        documents = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                    
                if file_path.endswith((".pdf", ".docx", ".txt", ".csv")):
                    # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                    loader = UnstructuredFileLoader(file_path)
                    loaded_documents = loader.load()

                    # Extend the main documents list with the loaded documents
                    documents.extend(loaded_documents)
       

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the assistant using the latest chat history
        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your files.")
