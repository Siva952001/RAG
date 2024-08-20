import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import os
import time
import json
import shutil
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from googletrans import Translator

MODEL_DIR = "C:/Users/DEV-037/.ollama/models/manifests/registry.ollama.ai/library"
FILE_DIR = 'files'
DATA_DIR = 'data'
SESSION_DIR = 'sessions'
translator = Translator()

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

def list_models(model_dir):
    return [model for model in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, model))]

def save_session(session_id):
    session_data = {
        "chat_history": st.session_state.chat_history,
        "processed_files": st.session_state.processed_files,
        "selected_model": st.session_state.selected_model
    }
    with open(os.path.join(SESSION_DIR, f"{session_id}.json"), 'w') as f:
        json.dump(session_data, f)

def load_session(session_id):
    with open(os.path.join(SESSION_DIR, f"{session_id}.json"), 'r') as f:
        session_data = json.load(f)
    st.session_state.chat_history = session_data.get("chat_history", [])
    st.session_state.processed_files = session_data.get("processed_files", {})
    st.session_state.selected_model = session_data.get("selected_model", None)

def clear_data():
    if os.path.exists(FILE_DIR):
        shutil.rmtree(FILE_DIR)
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(FILE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    st.session_state.chat_history = []
    st.session_state.processed_files = {}

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

# Session management UI
st.sidebar.header("Chat Session Management")
session_id = st.sidebar.text_input("Session ID", value="default")
if st.sidebar.button("Save Session"):
    save_session(session_id)
    st.sidebar.success("Session saved successfully")
if st.sidebar.button("Load Session"):
    load_session(session_id)
    st.sidebar.success("Session loaded successfully")
if st.sidebar.button("Clear"):
    clear_data()
    st.sidebar.success("Cleared files and data successfully")

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

st.title("Excel Chatbot")

st.sidebar.header("Upload Excel Files")
uploaded_files = st.sidebar.file_uploader("Upload your Excel files", type=['xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        file_path = os.path.join(FILE_DIR, uploaded_file.name)
        vectorstore_path = os.path.join(DATA_DIR, f"{uploaded_file.name}_vectorstore")

        if file_key in st.session_state.processed_files:
            st.sidebar.warning(f"File '{uploaded_file.name}' already exists. You can ask questions.")
        elif os.path.exists(vectorstore_path):
            st.sidebar.warning(f"It looks like the vector store for '{uploaded_file.name}' is already present in our records. You can continue your chat.")
            # Load the existing vector store
            st.session_state.vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=st.session_state.embedding_model)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            st.session_state.processed_files[file_key] = True
        else:
            try:
                with st.spinner("Analyzing your document..."):
                    # Save the uploaded file to the file directory
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Proceed with processing the file
                    xls = pd.ExcelFile(file_path)
                    all_content = ""
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        data = df.to_dict(orient='records')
                        content = "\n".join([str(row) for row in data])
                        all_content += content + "\n"

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=300,
                        length_function=len
                    )

                    splits = text_splitter.split_text(all_content)
                    documents = [Document(page_content=split) for split in splits]

                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=st.session_state.embedding_model,
                        persist_directory=vectorstore_path
                    )
                    st.session_state.vectorstore.persist()

                    st.session_state.processed_files[file_key] = True
                    st.sidebar.success(f"File '{uploaded_file.name}' processed successfully!")

                st.session_state.retriever = st.session_state.vectorstore.as_retriever()

            except PermissionError:
                st.error(f"Permission denied: Unable to write to {file_path}. Please check if the file is open or locked.")

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

st.subheader("Chat with Excel Files")
col1, col2 = st.columns([1, 2])

with col1:
    user_input = st.text_input("Ask a question about the Excel files:", key="user_input", placeholder="Type your question here...")
    if st.button("Submit"):
        if user_input:
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            st.markdown(f"**You:** {user_input}")

            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
                response_text = response['result']
                
                # Translate response to Tamil
                def safe_translate(text, dest_lang):
                    try:
                        translated_response_parts = translator.translate(text, dest=dest_lang)
                        if isinstance(translated_response_parts, list):
                            translated_text = ' '.join([part.text if part.text is not None else "" for part in translated_response_parts])
                        else:
                            translated_text = translated_response_parts.text if translated_response_parts.text is not None else ""
                        return translated_text
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
                        return text  # Fallback to the original text if translation fails

                translated_response = safe_translate(response_text, 'ta')
                
                # Display response with translation
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_text.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                chatbot_message = {
                    "role": "assistant", 
                    "message": response_text, 
                    "translated_message": translated_response
                }
                st.session_state.chat_history.append(chatbot_message)

with col2:
    st.subheader("Assistant's Response")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "assistant":
                original_content = message["message"]
                translated_content = message.get("translated_message", "")
                st.markdown(f"**Assistant (Original):** {original_content}")
                st.markdown(f"**Assistant (Tamil):** {translated_content}")

st.write("---")

st.subheader("Chat History")
for message in st.session_state.chat_history:
    role = message["role"]
    original_content = message["message"]
    translated_content = message.get("translated_message", "")
    st.markdown(f"**{role.capitalize()} (Original):** {original_content}")
    st.markdown(f"**{role.capitalize()} (Tamil):** {translated_content}")
