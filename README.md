# RAG (Retrieval-Augmented Generation) 系統

本專案為多種 RAG（檢索增強生成）流程的實作範例，基於 [LlamaIndex](https://github.com/jerryjliu/llama_index) 框架，結合 Hugging Face Embedding、Google Gemini API 等現代 NLP 技術，並以 Python 為主體。  
未來將陸續實作多種 RAG 變體，包含傳統 RAG、Corrective RAG (CRAG)、Self-RAG 及 LightRAG/Graph RAG。

---

## 目錄結構

- `RAG_llama_index.py`：主程式，示範如何用 LlamaIndex 串接 Hugging Face Embedding 與 Gemini LLM，並進行 RAG 查詢。
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

## 進階 RAG 變體（規劃中）

### 1. Corrective RAG (CRAG)

- **核心：品質評估。**
- 檢索後，先用 LLM 評估文檔與問題的相關性與可信度，給出分數。
- 若分數高，將文檔與問題傳給 LLM 生成答案；若分數低，僅用 LLM 內部知識回答。

### 2. Self-RAG

- **核心：自主決策。**
- 問題進來後，先問 LLM 是否需要檢索外部資料。
- 若需要，走傳統 RAG 流程；否則直接用 LLM 回答。

### 3. LightRAG / Graph RAG

- **核心：結合圖譜資料庫（如 Neo4j）與向量資料庫。**
- 先用 LLM 解析問題關鍵實體，查圖譜資料庫找關聯，再用這些實體去向量資料庫精確檢索，最後合併所有上下文給 LLM 生成答案。

---

## 目前進度

- [x] 傳統 RAG 流程（LlamaIndex + Hugging Face Embedding + Gemini LLM）
- [ ] Corrective RAG (CRAG)
- [ ] Self-RAG
- [ ] LightRAG / Graph RAG

---

## 環境變數說明

請將你的 API 金鑰等敏感資訊放在 `.env` 檔案中，例如：

```
GEMINI_API_KEY=你的_Gemini_API_KEY
```

