import streamlit as st
import base64
import json
import time
from uuid import uuid4
import requests
import os

USER_AVATAR = None
BOT_AVATAR = None
HOST_IP = os.getenv('HOST_IP', None)
API_BASE_URL = f"http://{HOST_IP}:8500" if HOST_IP is not None else "http://127.0.0.1:8500"
STREAM_RESPONSE = os.getenv("STREAM_RESPONSE", "true").lower() == "true"

def handle_answer(st, answer):
    if STREAM_RESPONSE:
    # Simulate streaming
        def stream_response():
            for word in answer.split(" "):
                yield word + " "
                time.sleep(0.05)
        displayed = st.write_stream(stream_response())
    else:
        st.markdown(answer)

def prepare_document_payload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return {
        "raw_document": encoded,
        "filename": filename,
        "session_id": session_id
    }


def prepare_query_payload(query: str, session_id: str, selected_docs: list) -> dict:
    return {
        "message": query,
        "session_id": session_id,
        "doc_ids": selected_docs,
        "role": "user",
    }


def fetch_session_conversation(session_id: str) -> list:
    """Fetch conversation history for a session from the backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/conversation"
        )
        if response.status_code == 200:
            return [
                {"role": entry["role"], "content": entry["message"]}
                for entry in response.json()
            ]
        else:
            st.toast(
                f"Could not load session history ({response.status_code})", icon="⚠️")
            return []
    except requests.exceptions.RequestException as e:
        st.toast(f"Could not reach backend: {e}", icon="⚠️")
        return []


def fetch_all_documents() -> dict:
    """Fetch all documents stored in the backend, returns {doc_id: filename}."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return response.json()
        else:
            st.toast(
                f"Could not load documents ({response.status_code})", icon="⚠️")
            return {}
    except requests.exceptions.RequestException as e:
        st.toast(f"Could not reach backend: {e}", icon="⚠️")
        return {}


def fetch_all_sessions() -> list[str]:
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []

def send_query(query: str, session_id: str, selected_docs: list) -> str:
    """Send a user query and return the assistant's response text."""
    payload = prepare_query_payload(query, session_id, selected_docs)
    try:
        response = requests.put(
            f"{API_BASE_URL}/sessions/{session_id}/conversation",
            json=payload,
        )
        if response.status_code == 200:
            return response.json()  # backend now returns a plain string
            #return response.json().get("response", "_(No response)_")
        else:
            return f"_(Backend error {response.status_code})_"
    except requests.exceptions.RequestException as e:
        return f"_(Could not reach backend: {e})_"


def switch_session(session_id: str):
    """Switch the active session and load its conversation from the backend."""
    st.session_state.session_id = session_id
    st.session_state.messages = fetch_session_conversation(session_id)
    # Restore uploaded docs for the session if tracked
    session_meta = st.session_state.all_sessions.get(session_id, {})
    st.session_state.uploaded_docs = session_meta.get("uploaded_docs", {})
    st.session_state.uploader_key += 1


def save_current_session_meta():
    """Persist lightweight metadata for the current session."""
    sid = st.session_state.session_id
    if sid not in st.session_state.all_sessions:
        st.session_state.all_sessions[sid] = {
            "label": f"Session {sid[:6]}",
            "created_at": time.strftime("%H:%M %d/%m/%y")
        }

# UI Config
st.set_page_config(
    page_title="RAG Group 23",
    page_icon="📚",
    layout="wide",
)

# ── Session state bootstrap ────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_docs" not in st.session_state:
    st.session_state.all_docs = fetch_all_documents()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}
    for sid in fetch_all_sessions():
        st.session_state.all_sessions[sid] = {
            "label": f"Session {sid[:6]}",
            "created_at": "—",
            "uploaded_docs": {},
        }

# Make sure the current session is always registered
save_current_session_meta()

session_id = st.session_state.session_id
short_id = session_id[:6]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Research Paper RAG")
    st.caption("Group 23")
    st.divider()

    # ── Active session ─────────────────────────────────────────────────────
    st.markdown("**Session**")
    st.code(short_id, language="text")

    if st.button("🆕 New session", use_container_width=True):
        new_id = str(uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        save_current_session_meta()
        st.rerun()

    st.divider()

    # ── Past sessions list ─────────────────────────────────────────────────
    st.markdown("**Sessions**")

    all_sessions = st.session_state.all_sessions
    if len(all_sessions) <= 1:
        st.caption("No other sessions yet.")
    else:
        st.markdown(
            """
            <style>
            div[data-testid="stVerticalBlock"] .session-list-container
                > div[data-testid="stVerticalBlock"] {
                max-height: 180px;
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(height=180, key="session_list_container"):
            for sid, meta in reversed(list(all_sessions.items())):
                is_active = sid == session_id
                label = f"{'🟢 ' if is_active else ''}{meta['label']}"
                help_text = f"Created {meta['created_at']}"
                btn_type = "primary" if is_active else "secondary"

                if st.button(
                    label,
                    key=f"sess_btn_{sid}",
                    use_container_width=True,
                    type=btn_type,
                    help=help_text,
                    disabled=is_active,
                ):
                    switch_session(sid)
                    st.rerun()

    st.divider()

    # ── Document upload ────────────────────────────────────────────────────
    st.markdown("**Documents**")
    uploaded_files = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        doc_payload = prepare_document_payload(
            file_bytes, uploaded_file.name, short_id
        )
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            response = requests.post(
                f"{API_BASE_URL}/documents",
                json=doc_payload,
            )
        if response.status_code == 200:
            st.session_state.all_docs = fetch_all_documents()  # refresh globally
            st.toast(f"{uploaded_file.name} uploaded!", icon="✅")
        else:
            st.toast(f"❌ Upload failed ({response.status_code})", icon="❌")

    if uploaded_files:
        st.session_state.uploader_key += 1
        st.rerun()

    # ── Document list (scrollable + selectable) ────────────────────────────
    if st.session_state.all_docs:
        with st.container(height=200):
            selected_doc_ids = []
            for doc_id, filename in st.session_state.all_docs.items():
                checked = st.checkbox(f"📄 {filename}", key=f"doc_{doc_id}")
                if checked:
                    selected_doc_ids.append(doc_id)
        st.session_state.selected_docs = selected_doc_ids
    else:
        st.caption("No documents uploaded yet.")

    st.divider()

# ── Main chat ──────────────────────────────────────────────────────────────
st.title("Research Paper RAG 📚")
st.caption("Ask questions about your uploaded research papers.")


st.session_state.messages = fetch_session_conversation(session_id)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(
            "Hi! Upload a paper from the sidebar and ask me anything about it."
        )

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

selected_docs = st.session_state.get("selected_docs", [])
uploaded_docs = st.session_state.get("all_docs", {})

if uploaded_docs and not selected_docs:
    st.warning(
        "No documents selected — check at least one in the sidebar for the model to use.", icon="📄")

query = st.chat_input("Ask a question about your documents...")

if query:
    # 1. Show and persist the user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    # 2. Get and show the assistant response
    selected_docs = st.session_state.get("selected_docs", [])
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            answer = send_query(query, session_id, selected_docs)
        handle_answer(st, answer)
        # #st.markdown(answer)
        
        # # Simulate streaming
        # def stream_response():
        #     for word in answer.split(" "):
        #         yield word + " "
        #         time.sleep(0.05)
        # displayed = st.write_stream(stream_response())

    # 3. Persist the assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})