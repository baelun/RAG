

#  STEP 1: 載入 PDF 並分段

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
def load_pdf(file_path):
    doc = fitz.open(file_path)
    paragraphs = []
    for page in doc:
        text = page.get_text()
        paragraphs.extend([p.strip() for p in re.split(r'(?<=[。！？!?])\s+|\n+', text) if p.strip()])
    return paragraphs



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

def search_similar(index,model,paragraphs,query, k=10):         # STEP 4: 查詢相關段落，返回前三相似的片段
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
#     prompt = f""" You need to answer the question in the sentence as same as in the  pdf content. . 
        # Given below is the context and question of the user.
        # context = {context}
        # question = {query}
        # if the answer is not in the pdf , answer "i donot know what the hell you are asking about"
        #  """

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response["choices"][0]["message"]["content"]

def greet(query,pdf_path="oreilly-technical-guide-understanding-etl.pdf" ):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paragraphs = load_pdf(pdf_path)
    embeddings = embedding_paragraph(model, paragraphs)
    index = create_faiss_index(embeddings)
    relevant = search_similar(index, model, paragraphs,query, k=10)
    context = "\n\n---\n\n".join(relevant)  # 每段加分隔線
    print("=== 查詢結果 ===\n")
    print(context.strip())
    return(context.strip())

    

def build_API():
    import gradio as gr
    # Use the gradio library to display a user interface for your user to interact with.
    pdf_path = "oreilly-technical-guide-understanding-etl.pdf" 
    demo = gr.Interface(
        fn=greet,
        inputs=gr.Textbox(lines=2, placeholder="輸入你的問題..."),
        outputs="text",
        title="📘 PDF 問答搜尋",
        description="輸入問題，我會幫你從 PDF 中找出最相關的段落"
    )

    # Launch the user demo - This can be done directly in your colab notebook. On your local notebook, you can also give a personalized localhost:port address.
    demo.launch(share=True,debug=False)



def main():
     
    # query = "ETL or ELT, which is better?"
    build_API()
    
    # answer = ask_gpt(query, context)
    # print("\nAnswer:\n", answer)



if __name__ == "__main__":
    main()



