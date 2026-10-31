import hashlib
import os
import re

import httpx
import streamlit as st

API_BASE_URL = os.getenv("PV_ASSIST_API_URL", "http://localhost:8001")
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="PV Assist", layout="wide")

st.title("PV Assist")
try:
    health_response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
    selected_provider = health_response.json().get("llm_provider", os.getenv("LLM_PROVIDER", "groq"))
except (httpx.HTTPError, ValueError):
    selected_provider = os.getenv("LLM_PROVIDER", "groq")
st.caption(f"Photovoltaic maintenance assistant | Answer provider: {selected_provider.title()}")

uploaded_file = st.file_uploader("Upload a PDF manual", type=["pdf"], key="pdf_uploader")
content_hash: str | None = None

if uploaded_file is not None and uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
    st.error(f"The PDF exceeds the {MAX_UPLOAD_SIZE_BYTES} byte upload limit.")
    uploaded_file = None

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    content_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    index_key = f"indexed_{content_hash}"
    if st.button("Index PDF", type="primary", disabled=bool(st.session_state.get(index_key))):
        try:
            with st.spinner("Uploading and indexing the PDF..."):
                response = httpx.post(
                    f"{API_BASE_URL}/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_bytes, "application/pdf")},
                    timeout=60.0,
                )
            if response.status_code == 200:
                payload = response.json()
                st.session_state[index_key] = payload["document_id"]
                st.success(f"Indexed {payload['document_name']} ({payload['chunks_indexed']} chunks)")
            else:
                st.error(f"Upload failed: {response.text}")
        except httpx.HTTPError as exc:
            st.error(f"Backend connection failed while indexing PDF: {exc}")
    document_id = st.session_state.get(index_key)
else:
    document_id = None

if st.session_state.get("answer_document_scope") != content_hash:
    st.session_state.pop("last_ask_result", None)
    st.session_state.pop("last_ask_question", None)
    st.session_state["answer_document_scope"] = content_hash

with st.form("ask_form"):
    question = st.text_input("Ask a question about the uploaded PDF")
    submitted = st.form_submit_button("Ask", type="primary", disabled=document_id is None)

if submitted:
    if not question.strip():
        st.warning("Enter a question before submitting.")
    else:
        try:
            with st.spinner("Retrieving sources and preparing an answer..."):
                response = httpx.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question, "document_id": document_id},
                    timeout=120.0,
                )
            if response.status_code == 200:
                st.session_state["last_ask_question"] = question
                st.session_state["last_ask_result"] = response.json()
            else:
                st.error(f"Answer request failed: {response.text}")
        except httpx.HTTPError as exc:
            st.error(f"Backend connection failed while asking: {exc}")

ask_result = st.session_state.get("last_ask_result")
if ask_result is not None:
    st.subheader("Answer")
    source_references = {
        source.get("source_id") or f"S{index}": f"{source['document_name']}, page {source['page_number']}"
        for index, source in enumerate(ask_result["sources"], start=1)
    }

    def format_source_reference(match: re.Match[str]) -> str:
        source_id = match.group(1)
        reference = source_references.get(source_id)
        return f"[{reference}]" if reference else f"[unverified source identifier: {source_id}]"

    display_answer = re.sub(r"\[(S[^\]\s]*)\]", format_source_reference, ask_result["answer"])
    st.markdown(display_answer)
    st.caption("Source identifiers are checked against retrieved passages; this does not verify factual support.")
    if ask_result["insufficient_context"]:
        st.info("The available passages did not provide sufficient evidence, so the assistant abstained.")
    if ask_result["sources"]:
        with st.expander("Retrieved supporting passages"):
            for source in ask_result["sources"]:
                st.markdown(
                    f"**{source['document_name']} — page {source['page_number']}**  \n"
                    f"Document ID: `{source['document_id']}`"
                )
                st.write(source["passage"])
