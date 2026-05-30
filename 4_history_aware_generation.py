from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
model = ChatOllama(model="qwen3:4b")

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""
Answer the user's question using ONLY the provided context.

Question:
{user_question}

Context:
{"\n".join([doc.page_content for doc in docs])}

Rules:
1. Give a direct answer.
2. Do NOT mention documents, context, sources, passages, or retrieval.
3. Do NOT say things like "Document 1 says", "According to the documents", or "The context mentions".
4. If the answer exists in the context, answer naturally.
5. If the answer is not present, reply exactly:
   "I don't have enough information to answer that question based on the provided information."
"""
    # combined_input = f"""Based on the following documents, please answer this question: {user_question}

    # Documents:
    # {"\n".join([f"- {doc.page_content}" for doc in docs])}

    # Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    # """
    
    # Step 4: Get the answer
    messages = [
        # SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    SystemMessage(
    content="""
You are a RAG assistant.

Answer questions using ONLY the provided context.

Never mention:
- documents
- context
- sources
- retrieved passages

Do not explain where the answer came from.

Give concise natural answers.

If the answer cannot be found, say:
"I don't have enough information to answer that question based on the provided information."
"""
)
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' or 'q' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit' or question.lower() == 'q':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()