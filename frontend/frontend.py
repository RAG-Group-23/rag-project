import streamlit as st
import base64
import time
from uuid import uuid4
import requests
import os
from urllib.parse import urlparse

USER_AVATAR = None
BOT_AVATAR = None
HOST_IP = os.getenv('HOST_IP', None)
API_BASE_URL = f"http://{HOST_IP}:8500" if HOST_IP is not None else "http://127.0.0.1:8500"
STREAM_RESPONSE = os.getenv("STREAM_RESPONSE", "false").lower() == "true"


def handle_answer(st, answer, stream_response=True):
    if stream_response:
        placeholder = st.empty()
        rendered_text = ""
        for word in answer.split():
            rendered_text += word + " "
            placeholder.markdown(rendered_text, unsafe_allow_html=True)
            time.sleep(0.05)
    else:
        st.rerun()
        st.markdown(answer, unsafe_allow_html=True)


def prepare_document_payload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return {"raw_document": encoded, "filename": filename, "session_id": session_id}


def prepare_query_payload(query: str, session_id: str, selected_docs: list) -> dict:
    return {"message": query, "session_id": session_id, "doc_ids": selected_docs, "role": "user"}


def fetch_session_conversation(session_id: str) -> list:
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/conversation")
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
    payload = prepare_query_payload(query, session_id, selected_docs)
    try:
        response = requests.put(
            f"{API_BASE_URL}/sessions/{session_id}/conversation", json=payload
        )
        if response.status_code == 200:
            return response.json()
        else:
            return f"_(Backend error {response.status_code})_"
    except requests.exceptions.RequestException as e:
        return f"_(Could not reach backend: {e})_"


def switch_session(session_id: str):
    st.session_state.session_id = session_id
    st.session_state.messages = fetch_session_conversation(session_id)
    session_meta = st.session_state.all_sessions.get(session_id, {})
    st.session_state.uploaded_docs = session_meta.get("uploaded_docs", {})
    st.session_state.uploader_key += 1


def save_current_session_meta():
    sid = st.session_state.session_id
    if sid not in st.session_state.all_sessions:
        st.session_state.all_sessions[sid] = {
            "label": f"Session {sid[:6]}",
            "created_at": time.strftime("%H:%M %d/%m/%y"),
        }


def do_delete_session(sid: str):
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions/{sid}")
        if response.status_code == 200:
            st.session_state.all_sessions.pop(sid, None)
            if st.session_state.session_id == sid:
                st.session_state.session_id = str(uuid4())
                st.session_state.messages = []
                st.session_state.uploader_key += 1
                save_current_session_meta()
            st.toast("Session deleted.", icon="🗑️")
        else:
            st.toast(
                f"Could not delete session ({response.status_code})", icon="⚠️")
    except requests.exceptions.RequestException as e:
        st.toast(f"Could not reach backend: {e}", icon="⚠️")


def do_delete_document(doc_id: str, filename: str):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        if response.status_code == 200:
            st.session_state.all_docs.pop(doc_id, None)
            selected = st.session_state.get("selected_docs", [])
            if doc_id in selected:
                selected.remove(doc_id)
                st.session_state.selected_docs = selected
            st.toast(f'"{filename}" deleted.', icon="🗑️")
        else:
            st.toast(
                f"Could not delete document ({response.status_code})", icon="⚠️")
    except requests.exceptions.RequestException as e:
        st.toast(f"Could not reach backend: {e}", icon="⚠️")


# ── Modal dialogs ──────────────────────────────────────────────────────────

@st.dialog("Delete session")
def confirm_delete_session_dialog(sid: str, label: str):
    st.write(f"Are you sure you want to delete **{label}**?")
    st.caption("This will permanently remove all messages in this session.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Delete", type="primary", use_container_width=True):
            do_delete_session(sid)
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Delete selected documents")
def confirm_delete_documents_dialog(doc_ids: list[str]):
    docs_to_delete = {
        doc_id: st.session_state.all_docs[doc_id]
        for doc_id in doc_ids
        if doc_id in st.session_state.all_docs
    }
    st.write(
        f"Are you sure you want to delete **{len(docs_to_delete)}** document(s)?")
    for name in docs_to_delete.values():
        st.caption(f"📄 {name}")
    st.caption("This will also remove all indexed content for these documents.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Delete", type="primary", use_container_width=True):
            for doc_id, filename in docs_to_delete.items():
                do_delete_document(doc_id, filename)
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── UI Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Group 23", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 350px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
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

if "show_url_indexer" not in st.session_state:
    st.session_state.show_url_indexer = False

if "indexing_in_progress" not in st.session_state:
    st.session_state.indexing_in_progress = False

if "pending_urls" not in st.session_state:
    st.session_state.pending_urls = []

save_current_session_meta()

session_id = st.session_state.session_id
short_id = session_id[:6]

# ── Blocking indexing state: runs the request, then clears itself ──────────
# This block must come before the sidebar and main UI so st.stop() halts
# everything else while indexing is in progress.
if st.session_state.indexing_in_progress:
    st.title("Research Paper RAG 📚")
    with st.spinner("Indexing documents, please wait…"):
        urls = st.session_state.pending_urls
        try:
            response = requests.post(
                f"{API_BASE_URL}/documents/url",
                json={"urls": urls, "session_id": short_id},
            )
            if response.status_code == 200:
                st.session_state.all_docs = fetch_all_documents()
                st.toast(f"Indexed {len(urls)} document(s)!", icon="✅")
            else:
                st.toast(
                    f"Failed ({response.status_code}): {response.json().get('detail', '')}",
                    icon="❌",
                )
        except requests.exceptions.RequestException as e:
            st.toast(f"Could not reach backend: {e}", icon="⚠️")
        finally:
            st.session_state.indexing_in_progress = False
            st.session_state.pending_urls = []
            st.session_state.show_url_indexer = False
    st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Research Paper RAG", unsafe_allow_html=True)
    st.caption("Group 23")
    st.divider()

    # ── Sessions ───────────────────────────────────────────────────────────
    st.markdown("**Sessions**", unsafe_allow_html=True)

    all_sessions = st.session_state.all_sessions

    if st.button("🆕 New session", use_container_width=True):
        new_id = str(uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        save_current_session_meta()
        st.rerun()

    with st.container(height=180, key="session_list_container"):
        if not all_sessions:
            st.caption("No sessions yet.")
        else:
            for sid, meta in reversed(list(all_sessions.items())):
                conversation = fetch_session_conversation(sid)
                if not conversation:
                    first_user_msg = "No messages yet."
                elif conversation[0]["role"] == "user":
                    first_user_msg = conversation[0]["content"]
                else:
                    first_user_msg = conversation[1]["content"] if len(
                        conversation) > 1 else "No messages yet."

                is_active = sid == session_id
                label = f"{'🟢 ' if is_active else ''}{first_user_msg}"
                help_text = f"Created {meta['created_at']}"
                btn_type = "primary" if is_active else "secondary"

                col_sess, col_del = st.columns([5, 1])
                with col_sess:
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
                with col_del:
                    if st.button("🗑", key=f"sess_del_{sid}", help="Delete session", use_container_width=True):
                        confirm_delete_session_dialog(
                            sid, f"session {sid[:6]}")

    st.divider()

    # ── Documents ──────────────────────────────────────────────────────────
    st.markdown("**Documents**", unsafe_allow_html=True)

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
            file_bytes, uploaded_file.name, short_id)
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            response = requests.post(
                f"{API_BASE_URL}/documents", json=doc_payload)
        if response.status_code == 200:
            st.session_state.all_docs = fetch_all_documents()
            st.toast(f"{uploaded_file.name} uploaded!", icon="✅")
        else:
            st.toast(f"❌ Upload failed ({response.status_code})", icon="❌")

    if uploaded_files:
        st.session_state.uploader_key += 1
        st.rerun()

    if st.session_state.all_docs:
        with st.container(height=200):
            selected_doc_ids = []
            for doc_id, filename in st.session_state.all_docs.items():
                display_name = filename if len(
                    filename) <= 22 else filename[:19] + "..."
                checked = st.checkbox(
                    f"📄 {display_name}", key=f"doc_{doc_id}", help=filename)
                if checked:
                    selected_doc_ids.append(doc_id)

        st.session_state.selected_docs = selected_doc_ids

        if selected_doc_ids:
            if st.button("🗑 Delete selected", use_container_width=True):
                confirm_delete_documents_dialog(selected_doc_ids)
    else:
        st.caption("No documents uploaded yet.")

    st.divider()

    if st.button("🔗 Index from URL", use_container_width=True):
        st.session_state.show_url_indexer = True
        st.rerun()

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False

# ── URL indexer inline UI (replaces the main chat area) ───────────────────
if st.session_state.show_url_indexer:
    st.title("Research Paper RAG 📚")
    st.subheader("🔗 Index documents from URLs")
    st.caption("Enter one or more URLs separated by commas or newlines.")

    raw = st.text_area(
        "URLs",
        placeholder="https://example.com/paper1.pdf, https://example.com/paper2.pdf",
        height=120,
    )

    col_index, col_cancel = st.columns(2)
    with col_index:
        if st.button("Index", type="primary", use_container_width=True):
            urls = [u.strip() for u in raw.replace(
                "\n", ",").split(",") if u.strip()]
            urls = [u for u in urls if is_valid_url(u)]
            if not urls:
                st.warning("Please enter at least one URL.")
            else:
                st.session_state.pending_urls = urls
                st.session_state.indexing_in_progress = True
                st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_url_indexer = False
            st.rerun()

    st.stop()  # prevent the chat UI from rendering while the form is open


# ── Main chat ──────────────────────────────────────────────────────────────
st.title("Research Paper RAG 📚")
st.caption("Ask questions about your uploaded research papers.")

st.session_state.messages = fetch_session_conversation(session_id)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(
            "Hi! Upload a paper from the sidebar and ask me anything about it.",
            unsafe_allow_html=True,
        )

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

selected_docs = st.session_state.get("selected_docs", [])
uploaded_docs = st.session_state.get("all_docs", {})

if uploaded_docs and not selected_docs:
    st.warning(
        "No documents selected — check at least one in the sidebar for the model to use.",
        icon="📄",
    )

selected_docs = st.session_state.get("selected_docs", [])
uploaded_docs = st.session_state.get("all_docs", {})
is_chat_disabled = len(selected_docs) == 0

query = st.chat_input(
    "Ask a question about your documents...", disabled=is_chat_disabled)

if query:
    save_current_session_meta()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query, unsafe_allow_html=True)

    strictly_selected_docs = [
        doc_id for doc_id in st.session_state.get("all_docs", {})
        if st.session_state.get(f"doc_{doc_id}") == True
    ]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            answer = send_query(query, session_id, strictly_selected_docs)
        handle_answer(st, answer, stream_response=STREAM_RESPONSE)

    st.session_state.messages.append({"role": "assistant", "content": answer})
