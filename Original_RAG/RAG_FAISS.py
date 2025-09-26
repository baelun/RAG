

#  STEP 1: 載入 PDF 並分段

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

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



def embedding_paragraph(model, paragraphs):   
    # STEP 2: 將段落轉成向量
    embeddings = model.encode(paragraphs)  #  對每段做 embedding
    return embeddings

def create_faiss_index(embeddings):
    # STEP 3: 建立 FAISS 向量索引，把 embeddings 加進 FAISS 向量資料庫
    dimension = embeddings.shape[1]  #在FAISS創造一個embeddings維度的大小
    index = faiss.IndexFlatIP(dimension)  #使用 L2距離(IndexHNSWFlat、IndexPQ...)
    index.add(np.array(embeddings))
    return index

def search_similar(index,model,paragraphs,query, k=3):         # STEP 4: 查詢相關段落，返回前三相似的片段
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    
    print("D,I=",D,I)
    for i, idx in enumerate(I[0]):
        print(f"相似度: {D[0][i]:.4f} 內容: {paragraphs[idx]}")
    return [paragraphs[i] for i in I[0]]

# STEP 5: 使用 OpenAI 回答問題
 
# import openai

# openai.api_key = "your-api-key"  # <-- 請填入你的 API 金鑰

# def ask_gpt(query, context):
#     prompt = f"""你是一個有用的助理。 請使用 context 去回答問題

#     Context:
#     {context}
    
#     Question: {query}
#     Answer:"""

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response["choices"][0]["message"]["content"]




def main():
    pdf_path = "/docs/file_deep_learning.pdf"
    chunk = load_pdf(pdf_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = "what deep learning allows?"
    paragraphs = load_pdf(pdf_path)
    embeddings = embedding_paragraph(model, chunk)
    index = create_faiss_index(embeddings)
    relevant = search_similar(index, model, paragraphs,query, k=3)
    
    context = "\n\n".join(relevant)
    print("\n=== 查詢結果 ===\n", context)
    
    # answer = ask_gpt(query, context)
    # print("\nAnswer:\n", answer)



if __name__ == "__main__":
    main()



