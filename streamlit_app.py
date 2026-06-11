import streamlit as st
import requests

# ============================================
# CONFIG
# ============================================
API_URL = "http://127.0.0.1:8000"  # FastAPI backend এর ঠিকানা

st.set_page_config(
    page_title="Smart Doc QA",
    page_icon="📄",
    layout="centered",
)

# ============================================
# SESSION STATE (Streamlit এর "memory")
# ============================================
# Streamlit প্রতি interaction এ পুরো script আবার চালায়।
# তাই messages আর upload status মনে রাখতে session_state লাগে।
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

# ============================================
# HEADER
# ============================================
st.title("📄 Smart Doc QA")
st.caption("Upload a PDF and ask questions — answers grounded in your document.")

# ============================================
# SIDEBAR — PDF Upload
# ============================================
with st.sidebar:
    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Indexing document... ⏳"):
                # FastAPI /upload কে call করি
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.doc_uploaded = True
                        st.session_state.messages = []  # নতুন doc = নতুন chat
                        st.success(f"✅ Indexed: {data['pages']} pages, {data['chunks']} chunks")
                    else:
                        st.error("Upload failed. Is the backend running?")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot reach backend. Run: uvicorn app:app --reload")

    st.divider()

    # Example questions — user কে guide করে
    st.subheader("💡 Example Questions")
    st.markdown(
        "- What skills does the candidate have?\n"
        "- How many years of experience?\n"
        "- Is he suitable for an ML role?"
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================
# CHAT DISPLAY — আগের সব message দেখাও
# ============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================
# CHAT INPUT — user প্রশ্ন করে
# ============================================
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.doc_uploaded:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            guide_msg = (
                "👋 I'd love to help! But first, please upload a PDF "
                "from the sidebar and click **Process Document**. "
                "Once it's indexed, I can answer questions about it."
            )
            st.markdown(guide_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": guide_msg}
            )
    else:
        # User এর message দেখাও ও সংরক্ষণ করো
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Backend কে জিজ্ঞেস করো
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": prompt},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources_used", 0)

                        st.markdown(answer)
                        st.caption(f"📎 Answer based on {sources} document section(s)")

                        # 
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.error("Failed to get an answer.")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot reach backend. Is it running?")