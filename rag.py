import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="MY RAG", page_icon="💜")

# ---------- Purple theme ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #1e1b4b, #6d28d9, #db2777);}
h1, p, label {color: white;}
.stButton>button {background: linear-gradient(90deg, #8b5cf6, #ec4899); color: white; border: 0;}
</style>
""", unsafe_allow_html=True)

st.title("💜 MY RAG")

# ---------- Login page ----------
if "api_key" not in st.session_state:
    st.write("Enter your OpenAI API key to start.")
    key = st.text_input("API Key", type="password")
    if st.button("Login"):
        if key.startswith("sk-"):
            st.session_state.api_key = key
            st.rerun()
        else:
            st.error("Invalid API key")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.session_state.api_key

# ---------- Sidebar ----------
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 150, 50)
top_k = st.sidebar.slider("Chunks to retrieve", 2, 10, 4)
if st.sidebar.button("Logout"):
    del st.session_state.api_key
    st.rerun()

# ---------- Read files ----------
def read_file(file):
    name = file.name.lower()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=name)
    temp.write(file.getvalue())
    temp.close()

    if name.endswith(".pdf"):
        docs = PyPDFLoader(temp.name).load()
    elif name.endswith(".docx"):
        docs = Docx2txtLoader(temp.name).load()
    elif name.endswith(".csv"):
        docs = CSVLoader(temp.name).load()
    else:
        docs = UnstructuredExcelLoader(temp.name).load()

    os.remove(temp.name)
    return docs

# ---------- Upload ----------
files = st.file_uploader("Upload files", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)

if not files:
    st.info("Upload a file to start chatting.")
    st.stop()

# build the database only when the files or chunk settings change
names = [f.name for f in files] + [chunk_size, chunk_overlap]
if st.session_state.get("names") != names:
    with st.spinner("Reading your files..."):
        all_docs = []
        for f in files:
            all_docs = all_docs + read_file(f)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(all_docs)
        st.session_state.db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    st.session_state.names = names
    st.session_state.messages = []

# ---------- Chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your files")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # find the best matching chunks
    results = st.session_state.db.similarity_search(question, k=top_k)
    context = ""
    for r in results:
        context = context + r.page_content + "\n\n"

    prompt = "Answer using only this context. If the answer is not there, say 'I could not find that in the documents.'\n\n"
    prompt = prompt + "Context:\n" + context + "\nQuestion: " + question

    llm = ChatOpenAI(model=model, temperature=temperature)
    answer = llm.invoke(prompt).contentr

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for r in results:
                st.caption(r.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})