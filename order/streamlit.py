import streamlit as st
import time
from text_analyze import process_file, translate_text, get_vectorstore, get_qa_chain
from .db import save_chat_to_db

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Session management UI
st.sidebar.header("Session Management")
session_id = st.sidebar.text_input("Session ID", value="default")
if st.sidebar.button("Save Session"):
    st.session_state.save_session(session_id)
    st.sidebar.success("Session saved successfully")
if st.sidebar.button("Load Session"):
    st.session_state.load_session(session_id)
    st.sidebar.success("Session loaded successfully")
if st.sidebar.button("Clear"):
    st.session_state.clear_data()
    st.sidebar.success("Cleared files and data successfully")

selected_model = st.selectbox("Select LLM Model", st.session_state.available_models)
if st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.vectorstore, st.session_state.qa_chain = get_vectorstore(selected_model)
    
st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload your files", type=['xlsx', 'xls', 'pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if not st.session_state.processed_files.get(uploaded_file.name):
            file_path = f"files/{uploaded_file.name}"
            process_file(uploaded_file, file_path, st.session_state.selected_model)
            st.session_state.processed_files[uploaded_file.name] = True
            st.sidebar.success(f"File '{uploaded_file.name}' processed successfully!")

st.write("---")

# Layout containers
chat_container = st.container()
input_container = st.container()

# Display Chat History
with chat_container:
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"**You:** {message['message']}")
        else:
            st.markdown(f"**Assistant:** {message['message']}")
            st.markdown(f"**Translated:** {message['translated_message']}")

# Input Field at the Bottom
with input_container:
    st.write("---")
    user_input = st.text_input("Ask a question about the files:", key="user_input", placeholder="Type your question here...")
    if st.button("Submit"):
        if user_input:
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            st.markdown(f"**You:** {user_input}")

            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
                response_text = response['result']
                translated_response = translate_text(response_text, 'ta')

                # Display response with translation
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_text.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                chatbot_message = {"role": "assistant", "message": response_text, "translated_message": translated_response}
                st.session_state.chat_history.append(chatbot_message)

                st.markdown(f"**Assistant:** {full_response}")
                st.markdown(f"**Translated:** {translated_response}")
        else:
            st.warning("Please enter a question before submitting.")
