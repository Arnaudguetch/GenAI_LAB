import pytest

# ======================
# Mocks
# ======================

class MockDocument:
    def __init__(self, content):
        self.content = content

class MockPDFLoader:
    """Simule le chargement de PDF"""
    def __init__(self, path):
        self.path = path

    def load(self):
        # Retourne des "documents" fictifs
        return [MockDocument("Contenu doc 1"), MockDocument("Contenu doc 2")]

class MockVectorstore:
    """Simule FAISS vectorstore"""
    def __init__(self, documents, embeddings=None):
        self.documents = documents

    def as_retriever(self):
        return self

    def get_relevant_documents(self, query):
        # Retourne des contenus simulés
        return [doc.content for doc in self.documents]

class MockLLM:
    """Simule un modèle de chat"""
    def invoke(self, prompt_dict):
        question = prompt_dict.get("question", "")
        return {"answer": f"Réponse simulée pour : {question}"}

class MockMemory:
    """Simule la mémoire de conversation"""
    def __init__(self):
        self.chat_memory = []

# ======================
# Tests
# ======================

def test_pdf_loader():
    loader = MockPDFLoader("dummy.pdf")
    docs = loader.load()
    assert len(docs) == 2
    assert docs[0].content == "Contenu doc 1"
    assert docs[1].content == "Contenu doc 2"

def test_vectorstore_retriever():
    docs = [MockDocument("doc A"), MockDocument("doc B")]
    vectorstore = MockVectorstore(docs)
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents("n'importe quelle question")
    assert results == ["doc A", "doc B"]

def test_conversational_chain_with_mock():
    llm = MockLLM()
    memory = MockMemory()
    
    # Simule la conversation
    user_question = "Qu'est-ce qu'un pipeline ML ?"
    response = llm.invoke({"question": user_question})
    memory.chat_memory.append((user_question, response["answer"]))
    
    assert len(memory.chat_memory) == 1
    question, answer = memory.chat_memory[0]
    assert question == user_question
    assert answer == f"Réponse simulée pour : {user_question}"
