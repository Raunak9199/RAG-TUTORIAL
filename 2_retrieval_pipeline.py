import time

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
# Configuration
# ============================================================

PERSIST_DIRECTORY = "db/chroma_db"

QUERY = "Who succeeded Ze'ev Drori as CEO in October 2008?"
# QUERY = "What was NVIDIA's first graphics accelerator called?"

# ============================================================
# Load Embeddings + Vector DB
# ============================================================

print("Loading vector database...")

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ============================================================
# Retriever
# ============================================================

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3,
#         # "fetch_k": 10
#     }
# )

# ============================================================
# Retrieve Documents
# ============================================================

retrieval_start = time.time()

relevant_docs = retriever.invoke(QUERY)

retrieval_time = time.time() - retrieval_start

print(f"\nUser Query: {QUERY}")
print(f"Retrieval Time: {retrieval_time:.2f} sec")

print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content[:500])

# ============================================================
# Build Context
# ============================================================

# context = "\n\n".join(
#     doc.page_content
#     for doc in relevant_docs
# )
combined_input = f"""Based on the following retrieved documents, please answer the question: {QUERY}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from the retrieved documents. If the answer is not contained in the documents, say "I don't have enough information to answer that question based on the provided documents and don't use outside knowledge".
"""
# ============================================================
# LLM
# ============================================================

model = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# ============================================================
# Prompt
# ============================================================

messages = [
    SystemMessage(content="You are a helpful RAG assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input)
]
# 4. Keep answers concise.
# 5. Do not explain your reasoning.
# ============================================================
# Generate Answer
# ============================================================

print("\n--- Generated Answer ---\n")

generation_start = time.time()

# response_chunks = []

# for chunk in model.stream(messages):
#     if chunk.content:
#         print(chunk.content, end="", flush=True)
#         response_chunks.append(chunk.content)

result = model.invoke(messages)

generation_time = time.time() - generation_start

# response_text = "".join(response_chunks)
print("Content Only:\n")
print(result.content)

print("\n")
print("=" * 60)

print(f"Retrieval Time : {retrieval_time:.2f} sec")
print(f"Generation Time: {generation_time:.2f} sec")
print(f"Total Time     : {retrieval_time + generation_time:.2f} sec")

print("=" * 60)



# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"


# old code

# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# from langchain_core.messages import SystemMessage, HumanMessage
# import time

# load_dotenv()

# persistent_directory = "db/chroma_db"

# # Load embeddings and vector store
# embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# db = Chroma(
#     persist_directory=persistent_directory,
#     embedding_function=embedding_model,
#     collection_metadata={"hnsw:space": "cosine"}  
# )

# # Search for relevant documents
# query = "What was NVIDIA's first graphics accelerator called?"
# # query = "How much did Microsoft pay to acquire GitHub?"

# retriever = db.as_retriever(search_kwargs={"k": 3})

# # retriever = db.as_retriever(
# #     search_type="similarity_score_threshold",
# #     search_kwargs={
# #         "k": 5,
# #         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
# #     }
# # )
# start = time.time()

# relevant_docs = retriever.invoke(query)
# print("Retrieval:", time.time() - start)

# print(f"User Query: {query}")
# # Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")



# combined_input = f"""Based on the following retrieved documents, please answer the question: {query}

# Documents:
# {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

# Please provide a clear, helpful answer using only the information from the retrieved documents. If the answer is not contained in the documents, say "I don't have enough information to answer that question based on the provided documents".
# """

# nstart = time.time()

# # Create the model
# model = ChatOllama(
#     model="qwen3:4b",
#     temperature=0
# )

# # Define a messages for the model
# messages = [
#     SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
#     HumanMessage(content=combined_input)
# ]

# # Generate a response
# result = model.invoke(messages)

# print("Generation:", time.time() - start)

# print("\n--- Generated Response: ---")

# print("Content Only:\n")
# print(result.content)
# # Synthetic Questions: 

# # 1. "What was NVIDIA's first graphics accelerator called?"
# # 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# # 3. "What was Microsoft's first hardware product release?"
# # 4. "How much did Microsoft pay to acquire GitHub?"
# # 5. "In what year did Tesla begin production of the Roadster?"
# # 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# # 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# # 8. "What was the original name of Microsoft before it became Microsoft?"