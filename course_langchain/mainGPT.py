#from langchain_community.chat_models import ChatOpenAI
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
import pickle
import tqdm
import time  

load_dotenv()

st.set_page_config(page_title="Chat avec PDF", page_icon="💬")

st.title("💬 Chat avec plusieurs documents PDF")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY manquant. Ajoute-le dans ton fichier .env")
    st.stop()

uploaded_files = st.file_uploader("Téléverse un ou plusieurs fichiers PDF pour commencer.", type="pdf", accept_multiple_files=True)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open("temp.pdf", "wb") as file:
            file.write(uploaded_file.read())
        
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        documents.extend(docs)
        
if documents:
        
        st.write(f"{len(documents)} documents chargés")

        print(f"{len(documents)} documents")
        print()

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        memory_file = "chat_memory.pkl"
        if os.path.exists(memory_file):
            with open(memory_file, "rb") as f:
                memory = pickle.load(f)
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
        def normalize_chat_history(history):
            normalized = []
            for msg in history:
                if isinstance(msg, tuple) and len(msg) == 2:
                    normalized.append(HumanMessage(content=msg[0]))
                    normalized.append(AIMessage(content=msg[1]))
                elif isinstance(msg, dict):
                    if msg.get("type") == "human":
                        normalized.append(HumanMessage(content=msg["data"]["content"]))
                    elif msg.get("type") == "ai":
                        normalized.append(AIMessage(content=msg["data"]["content"]))
                elif isinstance(msg, (HumanMessage, AIMessage)):
                    normalized.append(msg)
            return normalized


        #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = (
                memory.chat_memory.messages if memory.chat_memory.messages else []
            )
            #st.session_state.chat_history = []
            
        user_question = st.text_input("Pose ta question ici : ", key="user_input")
        
        if user_question:
            
            #question = "What is Machine Learning Pipeline ?"
            memory.chat_memory.messages = normalize_chat_history(st.session_state.chat_history)
            response = qa_chain.invoke({"question": user_question})
            #st.session_state.chat_history.append({"type": "human", "data": {"content": user_question}})
            #st.session_state.chat_history.append({"type": "ai", "data": {"content": response["answer"]}})

            st.session_state.chat_history.append((user_question, response['answer']))
            
            #memory.chat_memory.messages = st.session_state.chat_history
            memory.chat_memory.messages = normalize_chat_history(st.session_state.chat_history)
            with open(memory_file, "wb") as f:
                pickle.dump(memory, f)
                
        
        
        for msg in st.session_state.chat_history:
        
            if isinstance(msg, tuple):
                st.markdown(f"Vous : {msg[0]}")
                st.markdown(f"Assistant : {msg[1]}")
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    st.markdown(f"Vous : {msg['data']['content']}")
                elif msg.get("type") == "ai":
                    st.markdown(f"Assistant : {msg['data']['content']}")
            elif isinstance(msg, HumanMessage):
                st.markdown(f"Vous : {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"Assistant : {msg.content}")
                
        if st.button("Réinitialiser la mémoire du chat"):
            st.session_state.chat_history = []
            if os.path.exists(memory_file):
                os.remove(memory_file)
            st.success("Mémoire de chat réinitialisée")



        
        #for question, answer in st.session_state.chat_history:
        #    st.markdown(f"Vous :  {question}")
        #    st.markdown(f"assistant :  {answer}")
            
#else:
#    st.info(" Téléverse un ou plusieurs fichiers PDF pour commencer.")

        

#print()
#print(f"Response :\n\n {response['answer']}")

