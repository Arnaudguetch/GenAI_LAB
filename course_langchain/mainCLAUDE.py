from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("rapport_annuel.pdf")
documents = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key="...")
vectorstore = FAISS.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm=ChatAnthropic(model="claude-3-5-sonnet-2024-10", openai_api_key="...")

claude_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

question = "Quels sont les points clés du rapport de 2024 ?"
response = claude_chain({"question": question})

print(response["answer"])
