import os
import pickle
import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VERSION_FILE = "version.txt"


def load_version():
    
    """Lit la version actuelle depuis version.txt"""
    
    try:
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "V1.0"

APP_VERSION = load_version()


STORAGE_DIR = "rag_storage"
FAISS_DIR = os.path.join(STORAGE_DIR, APP_VERSION, "faiss_index")
MEMORY_FILE = os.path.join(STORAGE_DIR, APP_VERSION, "chat_memory.pkl")
PDF_CACHE_FILE = os.path.join(STORAGE_DIR, APP_VERSION, "pdf_cache.pkl")
DOCS_DIR = os.path.join(STORAGE_DIR, "uploaded_pdfs")

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


st.set_page_config(page_title="RAG Chat GPT ↔ LLaMA", layout="wide")
st.title(f"💬 ChatDocs — AGPTCS")


with st.sidebar:
    st.header("Configuration LLM")
    st.markdown(f"** Version actuelle : `{APP_VERSION}` stable **")

    llm_choice = st.selectbox("Choisir le LLM", ["GPT", "LLaMA", "ClauDE", "MistrAL"])

    gpt_models = ["gpt-4o", "gpt-5o"]
    llama_models = ["llama3.2:latest", "llama3.2:small"]

    if llm_choice == "GPT":
        model_name_gpt = st.selectbox("Choisir le modèle GPT", gpt_models, index=1)
        model_name_llama = None
    else:
        model_name_llama = st.selectbox("Choisir le modèle LLaMA", llama_models, index=0)
        model_name_gpt = None

    embeddings_choice = st.selectbox("Embeddings", ["OpenAIEmbeddings", "HuggingFace (local)"])

    reinitialize_index = st.button("Réindexer tous les PDFs")

st.markdown("---")
st.write("Téléverse tes PDFs pour créer un index de recherche :")
uploaded_files = st.file_uploader("Téléverser plusieurs PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
    st.success(f"{len(uploaded_files)} fichier(s) sauvegardé(s) dans `{DOCS_DIR}`")


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage



def get_embeddings(choice: str):
    if choice == "OpenAIEmbeddings":
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_faiss(doc_dir: str, embeddings):
    loader = DirectoryLoader(doc_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        st.warning("Aucun document PDF trouvé pour l’indexation.")
        return None

    vs = FAISS.from_documents(documents, embeddings)
    vs.save_local(FAISS_DIR)
    st.success(f"Index FAISS créé ({len(documents)} documents).")
    
    with open(PDF_CACHE_FILE, "wb") as f:
        pickle.dump(sorted(os.listdir(doc_dir)), f)

    return vs

def safe_load_faiss(embeddings):
    rebuild_required = reinitialize_index
    if os.path.exists(PDF_CACHE_FILE):
        with open(PDF_CACHE_FILE, "rb") as f:
            cached_files = pickle.load(f)
        current_files = sorted(os.listdir(DOCS_DIR))
        if current_files != cached_files:
            st.info("Nouveaux PDFs détectés. Reconstruction de l'index FAISS...")
            rebuild_required = True
    else:
        rebuild_required = True

    if rebuild_required:
        return build_or_load_faiss(DOCS_DIR, embeddings)

    if os.path.exists(FAISS_DIR) and os.listdir(FAISS_DIR):
        try:
            vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            st.info("Index FAISS chargé depuis le disque.")
            return vs
        except Exception as e:
            st.warning(f"Erreur de chargement : {e}. Reconstruction de l'index FAISS...")
            return build_or_load_faiss(DOCS_DIR, embeddings)
    else:
        return build_or_load_faiss(DOCS_DIR, embeddings)
    

def get_llm(choice: str, model_name_gpt: str, model_name_llama: str):
    if choice == "GPT":
        return ChatOpenAI(model_name=model_name_gpt, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    elif choice == "LLaMA":
        return ChatOllama(model=model_name_llama, temperature=0.0)
    return None

col1, col2, col3 = st.columns(3)
with col1: btn_build = st.button("Build / Load Index")
with col2: btn_clear_memory = st.button("Clear Memory")
with col3: btn_save_state = st.button("Save State")


embeddings = get_embeddings(embeddings_choice)
vectorstore = safe_load_faiss(embeddings)

if os.path.exists(MEMORY_FILE) and not btn_clear_memory:
    try:
        with open(MEMORY_FILE, "rb") as f:
            memory = pickle.load(f)
        st.info("Mémoire chargée depuis le disque.")
        st.session_state.chat_history = memory.chat_memory.messages
    except:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.chat_history = []
else:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.chat_history = []

if btn_clear_memory:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.chat_history = []
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    st.success("Mémoire réinitialisée.")


st.markdown("---")
st.header("💬 Conversation")

if vectorstore is None:
    st.warning("Pas d'index FAISS. Crée d’abord l’index.")
else:
    llm = get_llm(llm_choice, model_name_gpt, model_name_llama)
    if llm is None:
        st.error("LLM non disponible — vérifie la clé API ou le nom du modèle Ollama.")
    else:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory
        )

        user_question = st.text_input("Pose ta question ici :", key="user_input")
        submit = st.button("Envoyer")

        if submit and user_question:
            with st.spinner("🔎 Recherche + génération..."):
                try:
                    result = qa_chain({"question": user_question})
                    answer = result.get("answer") or result.get("result") or "Aucune réponse."
                except Exception as e:
                    answer = f"Erreur : {type(e).__name__} - {e}"

                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=answer))

                memory.chat_memory.messages = st.session_state.chat_history
                with open(MEMORY_FILE, "wb") as f:
                    pickle.dump(memory, f)

        for msg in st.session_state.chat_history[-20:]:
            if isinstance(msg, HumanMessage):
                st.markdown(f"** Toi : {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"** Assistant ({llm_choice}) : {msg.content}")
            st.markdown("---")


if btn_save_state:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = os.path.join(STORAGE_DIR, APP_VERSION, f"memory_{timestamp}.pkl")
    with open(state_file, "wb") as f:
        pickle.dump(memory, f)
    if vectorstore:
        vectorstore.save_local(FAISS_DIR)
    st.success(f"État sauvegardé pour {APP_VERSION} ({timestamp})")

st.caption("Assure-toi que OPENAI_API_KEY est définie pour GPT.")
