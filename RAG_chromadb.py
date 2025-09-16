
import fitz  # PyMuPDF
import os
#!-----------載入資料並切chunck

def load_pdf(file_path):
    doc = fitz.open(file_path)
    all_chunks = []
    chunk_size = 500 # 字元數為每個塊的大小
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text") # 使用 "text" 提取更結構化的文本

        # 將多個換行符替換為單個空格，以便更好地進行句子分割
        cleaned_text = ' '.join(text.splitlines()).strip()
        # 對於生產環境，請考慮使用 NLTK 的 sent_tokenize 或 spaCy。
        sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
        # 如果句點被 split 移除了，則重新添加句點
        sentences = [s + '.' if not s.endswith('.') else s for s in sentences]
        
        current_chunk_sentences = []
        current_chunk_length = 0

        
        for sentence in sentences:
            sentence_size = len(sentence)
            # 檢查添加下一個句子是否會超出塊大小
            # 即使一個句子比 chunk_size 長，也允許每個塊至少包含一個句子

            if current_chunk_length + sentence_size > chunk_size and current_chunk_sentences:
                all_chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [sentence]  #加sentence直到超過一個規定的 chunk size 
                current_chunk_length = sentence_size  # 累加sentence size
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_size + 1 # +1 是為了句子之間的空格
        # 添加當前塊中所有剩餘的句子
        if current_chunk_sentences:
            all_chunks.append(' '.join(current_chunk_sentences))

    doc.close() # 好的做法
    return all_chunks # 確保返回 all_chunks




# ----使用chromadb進行embedding向量轉換與儲存-----------------------------------------------------------
# pip install chromadb
import chromadb
from chromadb.utils import embedding_functions



def initial_chromadb(chunks, model):
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path="chroma_db")  # PersistentClient：將嵌入存儲在磁碟上以實現數據持久化


    collection = client.get_or_create_collection(name="pdf_chunks")
    # Generate unique IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True).tolist() 
    collection.add(
        documents=chunks,
        embeddings=embeddings,   #每個chunk的向量
        ids=ids,                #每個chunk的id
    )
    return collection


# 返回ids、embeddings、documents...metadatas、distances
def semantic_search(collection, query: str, n_results: int = 2):   # 回傳數量
    """Perform semantic search on the collection"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    # The results object contains documents, distances, and optionally embeddings/metadata
    retrieved_documents = results['documents'][0]
    distances = results['distances'][0]

    print("\n=== ChromaDB 查詢結果 ===")
    for i, doc in enumerate(retrieved_documents):
        print(f"相似度: {distances[i]:.4f} 內容: {doc}")
    return retrieved_documents

# ----------------------------------
# import os
# from openai import OpenAI

# # Initialize OpenAI client
# client = OpenAI()

# # Set your API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# def get_prompt(context: str, conversation_history: str, query: str):
#     """Generate a prompt combining context, history, and query"""
#     prompt = f"""Based on the following context and conversation history, 
#     please provide a relevant and contextual response. If the answer cannot 
#     be derived from the context, only use the conversation history or say 
#     "I cannot answer this based on the provided information."

#     Context from documents:
#     {context}

#     Previous conversation:
#     {conversation_history}

#     Human: {query}

#     Assistant:"""

#     return prompt

# def generate_response(query: str, context: str, conversation_history: str = ""):
#     """Generate a response using OpenAI with conversation history"""
#     prompt = get_prompt(context, conversation_history, query)

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",  # or gpt-3.5-turbo for lower cost
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0,  # Lower temperature for more focused responses
#             max_tokens=500
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error generating response: {str(e)}"



from sentence_transformers import SentenceTransformer

def main():
    pdf_path = "docs/file_deep_learning.pdf"
    chunks = load_pdf(pdf_path)
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Setup ChromaDB and add chunks
    chroma_collection = initial_chromadb(chunks, model)

    query = "what deep learning allows?"
    
    # Search for relevant chunks using ChromaDB
    relevant_chunks = semantic_search(chroma_collection, query, k=3)
    
    context = "\n\n".join(relevant_chunks)
    print("\n=== 傳遞給 LLM 的上下文 ===\n", context)
    
    # If you had the OpenAI part enabled:
    # answer = generate_response(query, context)
    # print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()

# https://blog.futuresmart.ai/building-rag-applications-without-langchain-or-llamaindex
