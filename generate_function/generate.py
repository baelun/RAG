
# --- 3. 輔助函數：生成模擬履歷內容 ---
def generate_resume_content():
    resume_text = []

    # --- 頁面 1：基本資訊與簡介 ---
    resume_text.append("="*80)
    resume_text.append("                     模擬履歷：陳明華 (Ming-Hua Chen)")
    resume_text.append("="*80)
    resume_text.append("\n**聯絡資訊**")
    resume_text.append("電話：+886 912 345 678")
    resume_text.append("Email：minghua.chen@example.com")
    resume_text.append("LinkedIn：linkedin.com/in/minghuachen_ai")
    resume_text.append("GitHub：github.com/minghuachen_dev")
    resume_text.append("個人網站：minghuachen.dev")

    resume_text.append("\n**專業簡介**")
    resume_text.append("一位經驗豐富、充滿熱情的**人工智慧與機器學習工程師**，擁有超過 8 年的產業經驗。專精於**自然語言處理 (NLP)**、**電腦視覺 (CV)** 和**推薦系統**的設計、開發與部署。擅長將前沿的 AI 研究應用於實際商業問題，提升產品性能與用戶體驗。具備領導跨功能團隊、從概念到實現完整專案生命週期的能力。熱衷於探索新技術，並致力於打造可擴展、高效能且具備商業價值的 AI 解決方案。")

    resume_text.append("\n**核心技能**")
    resume_text.append("- **程式語言**：Python (精通), Java, C++, R")
    resume_text.append("- **機器學習框架**：TensorFlow, PyTorch, Scikit-learn, Keras")
    resume_text.append("- **數據庫**：SQL (PostgreSQL, MySQL), NoSQL (MongoDB, Neo4j)")
    resume_text.append("- **雲平台**：AWS (EC2, S3, SageMaker, Lambda), Google Cloud Platform (GCP - AI Platform, BigQuery)")
    resume_text.append("- **大數據工具**：Spark, Hadoop, Kafka")
    resume_text.append("- **版本控制**：Git, SVN")
    resume_text.append("- **敏捷開發**：Scrum, Kanban")
    resume_text.append("- **專業領域**：自然語言處理 (NLP), 電腦視覺 (CV), 深度學習, 強化學習, 推薦系統, 數據分析, MLOps")

    resume_text.append("\n**工作經歷**")
    resume_text.append("---")
    resume_text.append("\n**高級人工智慧工程師** | 智創科技股份有限公司 (Innovate AI Solutions Inc.) | 台北, 台灣")
    resume_text.append("2021 年 8 月 – 至今")
    resume_text.append("- 負責設計並實施下一代智慧客服系統中的**語義理解模組**，利用 Transformer 模型 (BERT, GPT 系列) 將用戶查詢精確分類，將錯誤率降低 15%。")
    resume_text.append("- 主導開發**多模態產品推薦系統**，結合用戶行為數據、商品圖片和描述，利用 **CLIP** 及 Graph Neural Networks (GNN) 提升推薦準確度 20%，為公司帶來每月額外營收 5%。")
    resume_text.append("- 建立並維護基於 **MLOps 原則**的機器學習管道 (CI/CD)，實現模型訓練、部署、監控的自動化，縮短新模型上線時間 30%。")
    resume_text.append("- 帶領 3 人團隊，負責核心 AI 模型的研發與優化，提供技術指導與培訓。")

    # --- 頁面 2：更多工作經歷與專案 ---
    resume_text.append("\n" + "="*80) # 分頁符號示意
    resume_text.append("                     模擬履歷：陳明華 (續)")
    resume_text.append("="*80)

    resume_text.append("\n**人工智慧工程師** | 數據之星科技有限公司 (DataStar Tech Co.) | 新竹, 台灣")
    resume_text.append("2017 年 7 月 – 2021 年 7 月")
    resume_text.append("- 參與開發**智慧影像監控系統**中的**物件偵測模組**，使用 YOLOv4/YOLOv5 在邊緣設備上實現實時物件識別，準確率達 92%。")
    resume_text.append("- 參與構建針對金融詐欺偵測的**異常行為識別模型**，利用集成學習 (Ensemble Learning) 和時間序列分析，將詐欺偵測率提高 10%。")
    resume_text.append("- 負責數據預處理、特徵工程與模型評估，處理 TB 級別的非結構化與結構化數據。")
    resume_text.append("- 編寫技術文檔、設計開發規範，並向非技術背景的客戶解釋複雜的 AI 概念。")

    resume_text.append("\n**研究助理** | 國立台灣大學 資訊工程學系 | 台北, 台灣")
    resume_text.append("2016 年 9 月 – 2017 年 6 月")
    resume_text.append("- 參與**跨語言情感分析**專案，研究多語言詞向量模型在情感分類任務中的表現。")
    resume_text.append("- 協助數據採集、清洗與標註，並進行實驗結果分析。")

    resume_text.append("\n**學術背景**")
    resume_text.append("---")
    resume_text.append("\n**國立台灣大學 (National Taiwan University)** | 台北, 台灣")
    resume_text.append("資訊工程學系 (Department of Computer Science and Information Engineering)")
    resume_text.append("碩士 (Master of Science) | 2017 年 6 月畢業")
    resume_text.append("- 主修：人工智慧、機器學習、自然語言處理")
    resume_text.append("- 碩士論文：基於深度學習的中文醫學文本命名實體識別")
    resume_text.append("- 榮譽：書卷獎 (Dean's List), 兩次獲得系學術成果獎")

    resume_text.append("\n**國立成功大學 (National Cheng Kung University)** | 台南, 台灣")
    resume_text.append("資訊工程學系 (Department of Computer Science and Information Engineering)")
    resume_text.append("學士 (Bachelor of Science) | 2015 年 6 月畢業")
    resume_text.append("- 主修：演算法、數據結構、資料庫系統")
    resume_text.append("- 專題：基於影像處理的智慧交通流量監測系統")
    resume_text.append("- 榮譽：優秀畢業生")

    # --- 頁面 3：榮譽、專案與出版物 ---
    resume_text.append("\n" + "="*80) # 分頁符號示意
    resume_text.append("                     模擬履歷：陳明華 (續)")
    resume_text.append("="*80)

    resume_text.append("\n**個人專案 / 開源貢獻**")
    resume_text.append("---")
    resume_text.append("\n**RAG 知識圖譜聊天機器人** (https://github.com/minghuachen_dev/KG-RAG-Chatbot)")
    resume_text.append("- 開發一個結合 LlamaIndex、知識圖譜 (Neo4j) 和大型語言模型 (Gemini) 的 RAG 聊天機器人。")
    resume_text.append("- 實現了從非結構化文本自動抽取實體和關係構建知識圖譜的功能。")
    resume_text.append("- 提升了複雜查詢的回答精確度和可解釋性，並提供實時的上下文檢索。")

    resume_text.append("\n**基於 GAN 的圖像風格轉換器** (https://github.com/minghuachen_dev/GAN-Style-Transfer)")
    resume_text.append("- 實現了多種 GAN 模型的風格轉換，包括 CycleGAN 和 StarGAN。")
    resume_text.append("- 探討了不同損失函數和網路結構對生成圖像質量的影響。")

    resume_text.append("\n**榮譽與獎項**")
    resume_text.append("---")
    resume_text.append("- **AI Hackathon 2023 冠軍**：基於邊緣計算的智能安全帽識別系統 (2023年)")
    resume_text.append("- **Google Cloud AI Challenge 2020 優勝獎** (2020年)")
    resume_text.append("- **國立台灣大學 優秀畢業論文獎** (2017年)")

    resume_text.append("\n**出版物 / 研討會發表**")
    resume_text.append("---")
    resume_text.append("- Chen, M.-H., & Lin, Y.-J. (2017). **Deep Learning-based Named Entity Recognition for Traditional Chinese Medical Texts.** *Proceedings of the 29th Conference on Computational Linguistics and Speech Processing (ROCLING).* (碩士論文相關) ")
    resume_text.append("- Wang, P.-C., Chen, M.-H., & Hsieh, S.-L. (2019). **An Improved YOLO-based Object Detection System for Real-time Traffic Monitoring.** *International Journal of Computer Vision and Image Processing.* (工作期間發表) ")

    resume_text.append("\n**志工服務**")
    resume_text.append("---")
    resume_text.append("- **AI for Good 開源社區貢獻者**：參與開源項目，利用 AI 解決社會問題。")
    resume_text.append("- **社區科技普及志工**：定期舉辦 AI 基礎知識講座，向大眾推廣科技應用。")

    resume_text.append("\n**語言能力**")
    resume_text.append("---")
    resume_text.append("- 中文 (繁體)：母語")
    resume_text.append("- 英文：流利 (托福 iBT 105)")
    resume_text.append("- 日文：基礎")

    return "\n".join(resume_text)