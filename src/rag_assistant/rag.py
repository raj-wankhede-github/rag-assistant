import os
from dotenv import load_dotenv

# Core LangChain Imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda

# Document Handling & Embedding Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load variables directly from your root .env file
load_dotenv()

# Load documents from the docs/ folder
def load_documents():
    loaders = []
    # Resolving absolute paths allows uv run to target directories reliably
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    docs_path = os.path.join(base_dir, "src", "docs")
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print("Created docs/ folder. Add your PDFs and text files there.")
        return []
    
    # Load PDFs
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(os.path.join(docs_path, file)))
        elif file.endswith(".txt") or file.endswith(".md"):
            loaders.append(TextLoader(os.path.join(docs_path, file)))
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    return documents

# Split documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# Create vector store
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vectorstore

def format_docs(docs):
    """Helper to join retrieved documents together."""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chat chain with memory
def build_chain(vectorstore):
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.3)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 1. Pipeline Component: Reformulate context question if history exists
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


    # Sub-chain that turns a query + history into a standalone string question
    condense_question_chain = contextualize_q_prompt | llm | StrOutputParser()

    # Wrap inside RunnableLambda so LangChain allows streaming/piping cleanly
    get_search_query = RunnableLambda(
        lambda x: condense_question_chain if x.get("chat_history") else x["input"]
    )

    # 2. Pipeline Component: Generate the Final QA Answer
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Final combined execution flow using robust parameter assignments
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: get_search_query | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found. Add files to the src/docs folder and try again.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    vectorstore = create_vectorstore(chunks)
    print("Vector store created")
    
    chain = build_chain(vectorstore)
    print("\nRAG Assistant ready. Type your questions (type 'quit' to exit):\n")
    
    # Track the active chat session memory context
    chat_history = []

    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit", "q"]:
            break
        
        response_text = chain.invoke({"input": question, "chat_history": chat_history})
        print(f"\nAssistant: {response_text['answer']}\n")

        # Save historical turn arrays
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response_text))
        
        # Slicing window cap rule (last 10 turns = 20 entries)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

if __name__ == "__main__":
    main()
