import streamlit as st
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

st.title("📄 PDF Chat")
st.caption("Upload a PDF and ask questions in natural language. Powered by LangChain + Groq.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.success(f"✅ {data['pages']} pages · {data['chunks']} chunks")
            except httpx.ConnectError:
                st.error("Cannot reach the backend. Is it running on port 8000?")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.pdf_name:
        st.divider()
        st.markdown(f"**Active:** {st.session_state.pdf_name}")
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()

if not st.session_state.session_id:
    st.info("👈 Upload a PDF from the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        page_label = (
                            f"Page {src['page'] + 1}"
                            if isinstance(src["page"], int)
                            else f"Page {src['page']}"
                        )
                        st.markdown(f"**{page_label}**")
                        st.caption(src["content"])

    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/ask",
                        json={"session_id": st.session_state.session_id, "question": question},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data["sources"]:
                        with st.expander("📚 Sources"):
                            for src in data["sources"]:
                                page_label = (
                                    f"Page {src['page'] + 1}"
                                    if isinstance(src["page"], int)
                                    else f"Page {src['page']}"
                                )
                                st.markdown(f"**{page_label}**")
                                st.caption(src["content"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                    })
                except httpx.ConnectError:
                    err = "Error: Cannot reach the backend."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
