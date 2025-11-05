## ChatDocs – Application RAG multi-PDF : Développement d’un chatbot intelligent pour interroger des documents PDF.

#### Context :
> L'idée ici est de créer un chatbot Langchain + Streamlit (reponsive design) qui gère plusieurs documents PDF et mémoire persistance entre sessions.

#### Objectifs :
> 1. Téléverser un ou plusieurs PDFs (cours, rapports, manuels, etc.)
> 2. Répondre à des questions sur tous les documents (l'utilisateur pose une question -> le modèle va chercher les passages les plus pertinents dans les pdfs, puis génére une réponse)
> 3. Mémoire persistante : l'histoire est sauvegardé entre les sessions (via fichier local)
> 4. Compatible avec **GPT** ou **LLaMA** (Depuis l'interface tu peux passer dynamiquement de GPT (OpenAI) à LLaMA (OllaMA))
> 5. Définir une version de l'interface à chaque changement majeur
> 4. Le rendre portatible, sous forme d'application.

#### Bonnes pratiques :
> Comment fonctionne le code ?
>> Le code est contenu dans le fichier **script.py** à l'intérieur du dossier **course_langchain**. Pour commencer dans l'exécution du code, vous devez créer un fichier **.env** ou vous devez stocquer votre clé (OPENAI_KLEY_API="..." => afin d'eviter qu'il soit accessible à tous) que vous pouvez créer depuis votre compte **openai**, cette clé est obligatoire pour continuer le projet.
>> **QUELQUES NOTES :**
>> Aprés l'exécution du code, tu verras le dossier **rag_storage** dans ton éditeur de code bien évidemment ou se trouve :
>> - **faiss_index/** : sauvegarde locale
>> - **chat_memory.pkl** : mémoire persistance
>> - **uploaded_pdfs/** : PDFs uploadés 

 **Schéma détaillé de l'application RAG (Langchain) streamlit :**

```text
┌───────────────┐
│  Utilisateur  │
│  Pose une     │
│  question     │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────┐
│ Interface Streamlit           │
│ - Upload PDF                  │
│ - Choix LLM (GPT / LLaMA)     │
│ - Choix Embeddings            │
│ - Boutons Build / Clear / Save│
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Vérification des PDFs         |
│ - Compare PDF existants vs    │
│   cache (pickle)              │
│ - Si nouveaux PDFs détectés:  │
│   reconstruire FAISS          │
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  FAISS Index (Vector Store)   │
│ - Indexe les documents PDFs   │
│ - Stocke les embeddings       │
│ - Sauvegarde sur disque       │
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  RAG Chain (LangChain)        │
│ - Retriever : recherche       │
│   passages pertinents         │
│ - LLM : génère réponse        │
│   avec le contexte            │
└───────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Mémoire conversationnelle    │
│ - Stocke chat_history         │
│ - Persistante via pickle      │
│ - Permet multi-turn context   │
└───────┬───────────────────────┘
        │
        ▼
┌───────────────┐
│  Utilisateur  │
│  Reçoit la    │
│  réponse      │
└───────────────┘
```
#### Image de l'interface :

> Retrouver l'image dans le dossier image.