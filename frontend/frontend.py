import streamlit as st
import base64
import json
import time
from uuid import uuid4
import requests
import os

# TODO: Find some good user/bot asset pngs assets/bot.png
USER_AVATAR = None
BOT_AVATAR = None
HOST_IP = os.getenv('HOST_IP', None)
API_BASE_URL = f"http://{HOST_IP}:8000" if HOST_IP is not None else "http://127.0.0.1:8000"


def prepare_document_payload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return {
        "raw_document": encoded,
        "filename" : filename,
        "session_id" : session_id
    }


def prepare_query_payload(query: str) -> dict:
    return {
        "message": query
    }


def prepare_request(method: str, url: str, payload: dict) -> dict:
    return {
        "method": method,
        "url": url,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(payload, indent=2)
    }


def mock_send_query(query: str) -> dict:
    """Fake backend response."""
    return {
        "response": "I am not implemented yet"
    }


### UI Config
st.set_page_config(
    page_title="RAG Group 23",
    page_icon="📚",
    layout="wide",
)


# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

session_id = st.session_state.session_id[:6]

# Sidebar: session info, document upload, controls
with st.sidebar:
    st.markdown("### 📚 Research Paper RAG")
    st.caption("Group 23")
    st.divider()

    st.markdown("**Session**")
    st.code(session_id, language="text")
    if st.button("🆕 New session", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_docs = []
        st.rerun()

    st.divider()
    st.markdown("**Documents**")
    uploaded_files = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    # Multi-document uploud
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_docs:
            file_bytes = uploaded_file.read()
            doc_payload = prepare_document_payload(
                file_bytes, uploaded_file.name, session_id)

            with st.spinner(f"Uploading {uploaded_file.name}..."):
                response = requests.post(
                    f"{API_BASE_URL}/documents",
                    json=doc_payload,
                )

            if response.status_code == 200:
                st.session_state.uploaded_docs.append(uploaded_file.name)
                st.toast(f"{uploaded_file.name} uploaded!", icon="✅")
            else:
                st.toast(f"❌ Upload failed ({response.status_code})", icon="❌")

    if uploaded_files:
        st.session_state.uploader_key += 1
        st.rerun()


    if st.session_state.uploaded_docs:
        for name in st.session_state.uploaded_docs:
            st.markdown(f"- 📄 {name}")
    else:
        st.caption("No documents uploaded yet.")

    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main: chat
st.title("Research Paper RAG 📚")
st.caption("Ask questions about your uploaded research papers.")

# Empty-state hint
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(
            "Hi! Upload a paper from the sidebar and ask me anything about it. "
            "_(Responses are mocked for now.)_"
        )

# Display history
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input — st.chat_input only fires on submit, so the script no longer
# re-runs (and shows a response) on every keystroke.
query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    query_payload = prepare_query_payload(query)
    conversation_request = prepare_request(
        method="POST",
        url=f"/sessions/{session_id}/conversation",
        payload=query_payload,
    )

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            time.sleep(0.4)
            api_response = mock_send_query(query)
        st.markdown(api_response["response"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": api_response["response"],
    })