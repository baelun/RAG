# RAG (Retrieval-Augmented Generation) 系統

本專案為多種 RAG（檢索增強生成）流程的實作範例，基於 [LlamaIndex](https://github.com/jerryjliu/llama_index) 框架，結合 Hugging Face Embedding、Google Gemini API 等現代 NLP 技術，並以 Python 為主體。  
未來將陸續實作多種 RAG 變體，包含傳統 RAG、Corrective RAG (CRAG)、Self-RAG 及 LightRAG/Graph RAG。

---

## 目錄結構
- `original_RAG`：包含了各種非框架式與框架式實作，運用原生工具(如FAISS、Chromadb等vectorDB進行RAG實作)或是llamaindex。
- `graph_RAG`：包含了使用Neo4j GraphDB進行圖譜式資料儲存，提供LLM正確資訊進行結果產生。(內包含Docker實作)
- `README.md`：本說明文件。

---

## 進階 RAG 變體（規劃中）

### 1. Corrective RAG (CRAG)

- **核心：品質評估。**
- 檢索後，先用 LLM 評估文檔與問題的相關性與可信度，給出分數。
- 若分數高，將文檔與問題傳給 LLM 生成答案；若分數低，僅用 LLM 內部知識回答。

### 2. LightRAG / Graph RAG

- **核心：結合圖譜資料庫（如 Neo4j）與向量資料庫。**
- 先用 LLM 解析問題關鍵實體，查圖譜資料庫找關聯，再用這些實體去向量資料庫精確檢索，最後合併所有上下文給 LLM 生成答案。


### 3. Self-RAG

- **核心：自主決策。**
- 問題進來後，先問 LLM 是否需要檢索外部資料，所以實際上比較像是in-context的使用
- 若需要，走傳統 RAG 流程；否則直接用 LLM 回答。
- Self-RAG（Self-Retrieval-Augmented Generation）更像是一種架構上的決策或策略性設計選擇，而不一定是技術本身的根本差異。
- 這跟傳統的 RAG 最大的不同在於：模型本身先分析，再決定需不需要查資料，如果需要，會自己發出檢索指令（有點像思考過後才查資料）

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

實作心得：
1. 其實不同的RAG都大同小異，以目的性來說都是為了得到更準確的答案，所以技術上可能沒有最好，而是於使用者本身的準確度需求或是廣度需求，去選擇合適的RAG方法。需求細分的話可能有以下幾種：
- 使用者的資訊密度需求（淺 vs 深）
- 回答準確性 vs 資源成本
- latency vs recall
- 回答是否需要多來源交叉驗證

2. 在VectorDB上，主要差異都會是性能差異跟資源需求(for local or cloud)。
也可以細分成以下幾種：
- Pinecone：雲端為主、支援高可用與 metadata filter，適合商用但成本高(適合用來測試pipeline)
- Weaviate：開源、可 local，功能豐富、支援 hybrid search
- Faiss：meta 開源，超輕量、可完全 local，但缺乏索引管理功能（production 用要包裝一層）
- Milvus / Qdrant：偏 infra 級、適合需要自管 VectorDB 的場景(for公司內部可用)