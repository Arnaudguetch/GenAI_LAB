from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

loader = TextLoader("notes.txt")
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llama_chain = ConversationalRetrievalChain.from_llm(
    llm=Ollama(model="llama3"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

question = "Quels sujets sont abordés dans mes notes ?"
response = llama_chain({"question": question})

print(response["answer"])
