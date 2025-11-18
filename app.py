import streamlit as st
from backend import extract_text_from_pdf, chunk_text, build_faiss_index, search_faiss
from utils import save_uploaded_file
from model import load_granite_model, ask_model

# -------------------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------------------
st.set_page_config(page_title="StudyMate - AI PDF Assistant", layout="wide")

st.title("📘 StudyMate - AI-Powered PDF Academic Assistant")
st.write("Upload your PDFs and ask any question. StudyMate will answer from your material.")

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = load_granite_model()

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# -------------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Academic PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

process_btn = st.button("Process PDFs")

# -------------------------------------------------------------
# PROCESS PDFs
# -------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF!")
    else:
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.embeddings = None

        all_text = ""

        with st.spinner("Extracting and processing PDFs..."):
            for file in uploaded_files:
                try:
                    path = save_uploaded_file(file)
                    text = extract_text_from_pdf(path)

                    if text.strip():
                        all_text += text + "\n\n"
                    else:
                        st.warning(f"No text extracted from: {file.name}")

                except Exception as e:
                    st.warning(f"Skipping unreadable file: {file.name} ({e})")

        # No extracted text
        if not all_text.strip():
            st.error("No valid text found in uploaded PDFs.")
        else:
            chunks = chunk_text(all_text)
            index, embeddings = build_faiss_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embeddings = embeddings

            st.success("PDFs processed successfully!")

st.divider()

# -------------------------------------------------------------
# ASK QUESTION
# -------------------------------------------------------------
question = st.text_input("Ask a question from your study materials")

if st.button("Get Answer", type="primary"):
    if st.session_state.index is None:
        st.error("Please upload and process PDFs first!")
    else:
        with st.spinner("Searching materials..."):
            results = search_faiss(
                question,
                st.session_state.chunks,
                st.session_state.index,
                top_k=5
            )

            context = "\n\n".join(results)

        with st.spinner("Generating answer..."):
            answer = ask_model(st.session_state.model, question, context)

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📄 References from your PDFs")
        for i, ref in enumerate(results):
            with st.expander(f"Reference {i + 1}"):
                st.write(ref)
