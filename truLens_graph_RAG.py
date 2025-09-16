# (假設你已經設定好 LlamaIndex 的 Settings: LLM, Embedding Model)
# (假設你已經有一個 KnowledgeGraphIndex，並從中取得了 query_engine)

from trulens_eval import Tru
from trulens_eval import Feedback
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.prompts import GroundednessMeasures
from trulens_eval import Select
from trulens_eval.app import App

# 初始化 TruLens
tru = Tru()

# 創建評估回饋函數
# 這裡使用預設的 Groundedness 和 Answer Relevancy
f_groundedness = (
    Feedback(Groundedness(groundedness_measures=GroundednessMeasures()).coherence)
    .on(Select.RecordCalls.llm_output) # 評估 LLM 輸出
    .on_context(Select.RecordCalls.retriever_output.node_contents) # 檢索到的上下文
    .aggregate(Groundedness().grounded_statements_aggregator)
)

f_answer_relevance = (
    Feedback(tru.relevance_with_cot_reasons) # 使用 tru.relevance_with_cot_reasons 作為一個通用相關性評估器
    .on_query() # 評估查詢
    .on_llm_output() # 評估 LLM 輸出
)

# 假設這是你的 Graph RAG query_engine
# query_engine = my_graph_rag_index.as_query_engine()

# 包裝你的查詢引擎以進行評估
tru_recorder = App(query_engine, tru=tru, feedback_functions=[f_groundedness, f_answer_relevance])

# 執行查詢，並會自動記錄評估結果
with tru_recorder as recording:
    response = query_engine.query("你的圖譜相關問題")
    print(response)

# 查看評估結果
tru.run_dashboard() # 在 Colab 中，這會啟動一個可視化界面