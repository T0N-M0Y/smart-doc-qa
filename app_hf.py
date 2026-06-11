import streamlit as st
import os
import tempfile

from src.loader import load_pdf, split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore
from src.rag import answer_with_memory

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="Smart Doc QA", page_icon="📄", layout="centered")

# ============================================
# CACHED RESOURCES (একবার load হবে)
# ============================================
@st.cache_resource
def load_embedding_model():
    return get_embedding_model()

embedding_model = load_embedding_model()

# ============================================
# SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ============================================
# HEADER
# ============================================
st.title("📄 Smart Doc QA")
st.caption("Upload a PDF and ask questions — answers grounded in your document.")

# ============================================
# SIDEBAR — Upload
# ============================================

with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    # Button সবসময় দেখাই — file না থাকলে disabled
    process_clicked = st.button(
        "Process Document",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    if process_clicked and uploaded_file is not None:
        with st.spinner("Indexing document... ⏳"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            docs = load_pdf(tmp_path)
            chunks = split_documents(docs)
            st.session_state.vectorstore = create_vectorstore(
                chunks, embedding_model
            )
            st.session_state.messages = []
            os.unlink(tmp_path)

            st.success(f"✅ Indexed: {len(docs)} pages, {len(chunks)} chunks")

    st.divider()
    st.subheader("💡 Example Questions")
    st.markdown(
        "- What skills does the candidate have?\n"
        "- Summarize this document\n"
        "- How many years of experience?"
    )
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================
# CHAT DISPLAY
# ============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================
# CHAT INPUT
# ============================================
if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.vectorstore is None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            guide = (
                "👋 Please upload a PDF from the sidebar and click "
                "**Process Document** first. Then I can answer questions about it."
            )
            st.markdown(guide)
            st.session_state.messages.append({"role": "assistant", "content": guide})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                # History বানাই (greeting/assistant turn বাদে)
                history = []
                msgs = st.session_state.messages[:-1]
                for i in range(0, len(msgs) - 1, 2):
                    if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                        history.append({
                            "question": msgs[i]["content"],
                            "answer": msgs[i+1]["content"],
                        })

                answer, sources = answer_with_memory(
                    st.session_state.vectorstore, prompt, history
                )
                st.markdown(answer)

                if sources:
                    with st.expander(f"📎 Sources ({len(sources)} sections)"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Section {i}** (page {page})")
                            st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )