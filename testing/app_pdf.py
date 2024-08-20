import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time
import tempfile

MODEL_DIR = "C:/Users/DEV-037/.ollama/models/manifests/registry.ollama.ai/library"
FILE_DIR = 'files'
DATA_DIR = 'data'

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def list_models(model_dir):
    return [model for model in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, model))]

available_models = list_models(MODEL_DIR)

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

selected_model = st.selectbox("Select LLM Model", available_models)

if 'llm' not in st.session_state or st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model=selected_model,
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )
    st.session_state.embedding_model = OllamaEmbeddings(model=selected_model)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type='pdf')

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if file_key not in st.session_state.processed_files:
        with st.spinner("Analyzing your document..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, dir=FILE_DIR) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=st.session_state.embedding_model
            )
            st.session_state.vectorstore.persist()

            
            st.session_state.processed_files[file_key] = True

          
            os.remove(temp_file_path)

    #
    if 'vectorstore' in st.session_state:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()

        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )

st.write("---")


st.subheader("Chat with PDF")
col1, col2 = st.columns([1, 2])

with col1:
    user_input = st.text_input("Ask a question about the PDF:", key="user_input", placeholder="Type your question here...")
    if st.button("Submit"):
        if user_input:
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            st.markdown(f"**You:** {user_input}")

            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)

with col2:
    st.subheader("Assistant's Response")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "assistant":
                content = message["message"]
                st.markdown(f"**Assistant:** {content}")

st.write("---")

st.subheader("Chat History")
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["message"]
    st.markdown(f"**{role.capitalize()}:** {content}")
