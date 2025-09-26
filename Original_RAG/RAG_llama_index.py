# -*- coding: utf-8 -*-
"""
LlamaIndex RAG 範例 (Colab) - 重構成 Main 形式

這個筆記本將示範如何在 Colab 環境中，
利用 LlamaIndex 結合 Hugging Face 的 BGE-M3 Embedding 模型
進行檢索增強生成 (RAG)。
所有功能都包裝在 main 函數中。
"""

# --- 1. 安裝必要的套件 ---
print("--- 1. 安裝必要的套件 ---")
# 優先且強制安裝一個已知兼容的 transformers 版本
# !pip install -q --force-reinstall transformers==4.35.0 # 這個版本通常兼容
# !pip install -q sentence-transformers
# !pip install -q llama-index
# !pip install -q llama-index-embeddings-huggingface
# !pip install -q llama-index-llms-gemini
# !pip install -q python-decouple
# !pip install -q trulens-eval # 雖然這裡沒有使用，但保留以符合你的原始安裝

print("套件安裝完成。")

# --- 2. 導入必要的模組 ---
# 確保所有導入都在函數外部，避免重複導入
import os
from decouple import config
import generate_function.generate as generate_resume_content
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# 不需要 SimpleFileNodeParser，因為我們透過 Settings 控制 chunking
# from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine



# --- 4. 主函式 ---
def main():
    # --- 4.1. 配置 LlamaIndex Settings (Embedding Model & LLM) ---
    print("\n--- 4.1. 配置 LlamaIndex Settings ---")

    # 設定 Embedding 模型 (Hugging Face BGE-M3)
    print("設定 Embedding 模型: BAAI/bge-m3...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu", # 或 "cuda" 如果有 GPU
        embed_batch_size=8,
    )
    Settings.embed_model = embed_model
    print("Embedding Model 設定完成。")

    # 設定 LLM (Google Gemini 1.5 Flash)
    print("設定 LLM: Google Gemini 1.5 Flash...")
    try:
        # 從 .env 讀取 GEMINI_API_KEY
        gemini_api_key = config('GEMINI_API_KEY', default=None)
        if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
            raise ValueError("請在 .env 檔案中設定 'GEMINI_API_KEY'，或在程式碼中替換為你的金鑰。")
    except Exception as e:
        print(f"API 金鑰載入失敗: {e}")
        gemini_api_key = None

    if gemini_api_key:
        llm = Gemini(
            model="models/gemini-1.5-flash",
            api_key=gemini_api_key,
        )
        Settings.llm = llm
        print("LLM 配置完成。")
    else:
        print("WARN: 未能成功載入 Gemini API 金鑰，LLM 功能可能受限或無法使用。")

    # 設定節點解析 (chunking) 的參數
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
    print(f"節點切割參數設定: chunk_size={Settings.chunk_size}, chunk_overlap={Settings.chunk_overlap}")

    # --- 4.2. 準備範例數據 ---
    print("\n--- 4.2. 準備範例數據 (生成模擬履歷) ---")
    data_dir = "/content/dataFiles"
    os.makedirs(data_dir, exist_ok=True)

    # 生成履歷內容並保存
    resume_output = generate_resume_content()
    resume_file_path = os.path.join(data_dir, "陳明華_模擬履歷.txt")
    with open(resume_file_path, "w", encoding="utf-8") as f:
        f.write(resume_output)
    print(f"模擬履歷已生成並保存至 '{resume_file_path}'")

    # --- 4.3. 載入文件 ---
    print("\n--- 4.3. 載入文件 ---")
    # 注意: SimpleDirectoryReader 預設會讀取指定路徑下所有文件。
    # 確保 /content/dataFiles 只包含你想要處理的文件，否則可能讀取到不必要的檔案。
    documents = SimpleDirectoryReader(input_dir=data_dir).load_data(show_progress=True)
    print(f"總共載入 {len(documents)} 份文件。")

    # --- 4.4. 創建向量儲存索引 ---
    print("\n--- 4.4. 創建向量儲存索引 ---")
    # 直接從 documents 創建索引。LlamaIndex 會自動使用 Settings 中的 chunking 和 embedding 配置。
    print("從 documents 創建索引中 (這將自動進行 chunking 和 embedding)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("向量儲存索引創建完成。")

    # --- 4.5. 建立檢索器和查詢引擎 ---
    print("\n--- 4.5. 建立檢索器和查詢引擎 ---")
    retriever = index.as_retriever(similarity_top_k=2)
    print("檢索器建立完成。")

    # 建立查詢引擎。確保這裡只呼叫一次 `as_query_engine`。
    # 如果想指定檢索器，就傳入 `retriever=retriever`
    query_engine = index.as_query_engine(retriever=retriever)
    print("查詢引擎建立完成。")

    # --- 4.6. 執行 RAG 查詢 ---
    print("\n--- 4.6. 執行 RAG 查詢 ---")

    query_text = "陳明華曾參與過的專案與最厲害的經歷是?"
    print(f"\n問題: {query_text}")
    response = query_engine.query(query_text)
    print("回答:")
    print(response)

    # --- 4.7. 測試流式回應 ---
    print("\n--- 4.7. 測試流式回應 ---")
    # 如果你已經在 query_engine 初始化時指定了 retriever，這裡不用再指定
    streaming_query_engine = index.as_query_engine(retriever=retriever, streaming=True)
    streaming_query = "陳明華在職場上的職責與貢獻有哪些？"
    print(f"\n流式問題: {streaming_query}")
    streaming_response = streaming_query_engine.query(streaming_query)
    print("流式回答:")
    streaming_response.print_response_stream()

    print("\n\nRAG 範例執行完畢。")

# --- 5. 執行主函式 ---
if __name__ == "__main__":
    main()