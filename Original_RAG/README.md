
## 目錄結構


- `RAG_llama_index.py`：使用 LlamaIndex 串接 Hugging Face Embedding 與 Gemini LLM，並進行 RAG 查詢。
- `RAG_FAISS.py`：使用自建構VectorDB，創建本地式DB進行文本的embedding儲存與LLM生成使用。
- `RAG_FAISS_api.py`：使用gradio工具進行可視化測試。
- `RAG_chromadb.py`：使用chromadb自建構VectorDB，創建本地式DB進行文本的embedding儲存與LLM生成使用。
- `generate_function/`：自訂履歷內容生成模組。
- `.env`：環境變數檔，儲存 API 金鑰等敏感資訊。
- `README.md`：本說明文件。

---

## 快速開始

1. **安裝依賴套件**
   ```sh
   pip install -r requirements.txt
   ```
   或根據程式碼註解安裝：
   ```sh
   pip install transformers==4.35.0 sentence-transformers llama-index llama-index-embeddings-huggingface llama-index-llms-gemini python-decouple trulens-eval
   ```

2. **設定環境變數**
   - 在專案根目錄建立 `.env` 檔案，內容如下：
     ```
     GEMINI_API_KEY=你的_Gemini_API_KEY
     ```

3. **執行主程式**
   ```sh
   python RAG_llama_index.py
   ```

---

## 傳統 RAG (Retrieval-Augmented Generation) 流程

這是最基本的 RAG 流程，也是其他所有 RAG 變體的基礎。

1. **使用者輸入節點 (Start)**：接收使用者輸入的問題。
2. **嵌入節點 (OpenAI 或 Cohere)**：將問題轉換成向量。
3. **向量資料庫檢索節點 (Qdrant 或 Pinecone)**：用問題向量在資料庫中搜尋最相似的文檔塊。
4. **LLM 節點 (OpenAI 或 Google Gemini)**：將原始問題和檢索到的文檔作為上下文，發送給 LLM 進行生成。
5. **回傳結果節點**：將生成的答案回傳給使用者。

此流程簡單直接，適用於大部分基礎問答場景。

---

## 環境變數說明

請將你的 API 金鑰等敏感資訊放在 `.env` 檔案中，例如：

```
GEMINI_API_KEY=你的_Gemini_API_KEY
```

