# Open LLM Gateway - OpenAI 相容 API 伺服器 (多模型支援)

這是一個使用 FastAPI 建置的 Python 伺服器，提供與 OpenAI API 相容的介面，支援多種 AI 服務，包括開源和閉源的大型語言模型 (LLM) 以及文本嵌入 (Embedding) 功能。透過此伺服器，您只需設定一個 `base_url` 和 `api_key`，即可使用標準的 OpenAI API 或 CURL 請求來存取各家 LLM 供應商和嵌入模型。

## 1. 功能概述

### 1.1 主要特性
- **API 相容性**：與 OpenAI API 格式高度相容，方便現有專案無縫遷移。
- **多模型支援**：整合多種主流 LLM 供應商及本地模型，只需修改模型名稱即可切換。
- **文本嵌入 (Embedding)**：提供高效的文本嵌入功能，支援多種來源。
- **聊天補全 (Chat Completion)**：提供強大的聊天對話功能，支援串流與非串流模式。
- **串流輸出 (Streaming)**：所有聊天模型供應商均支援串流回應，可即時獲取生成內容。

### 1.2 支援的模型
本伺服器支援從開源到閉源的各種模型，涵蓋以下供應商：
- **開源 LLM 與嵌入模型**：
  - **Ollama**：支援所有本地運行的聊天和嵌入模型，例如 `ollama/llama3` 和 `ollama/nomic-embed-text`。
  - **Hugging Face**：支援所有相容模型，例如聊天模型 `huggingface/meta-llama/Llama-3-8B-Instruct` 和嵌入模型 `huggingface/bge-large-zh-v1.5`。
- **閉源 LLM 與嵌入模型**：
  - **OpenAI**：例如 `openai/gpt-4`, `openai/gpt-3.5-turbo`, `text-embedding-3-small`。
  - **Google Gemini**：例如 `gemini/gemini-1.5-pro-latest`, `models/embedding-001`。
  - **Anthropic Claude**：例如 `claude/claude-3-opus-20240229`, `claude/claude-3-haiku-20240307`。
  - **DeepSeek**：例如 `deepseek/deepseek-chat`。
  - **MinMax**：例如 `minmax/abab6-chat`。

### 1.3 附加功能
- **API 金鑰管理介面**：提供一個管理介面，允許您建立、刪除提供給他人使用的 API 金鑰，並監控他們的使用量，增強安全性和可追蹤性。
- **聊天示範頁面**：提供一個聊天頁面，示範如何應用此 Open LLM Gateway 進行即時對話。

## 2. 安裝與設定

### 2.1 環境需求
- Python 3.8 或更高版本
- `pip` 套件管理器
- 網際網路連線 (用於下載模型或連接雲端 API 服務)

### 2.2 安裝步驟
1. **複製專案**：
   將 `main.py`, `requirements.txt`, `README.md` 和 (可選的) `env.example` (需自行複製為 `.env`) 檔案放在您的專案目錄中。
2. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

### 2.3 API 金鑰設定
將 `env.example` 檔案複製一份並重新命名為 `.env`。然後，在 `.env` 檔案中設定您需要使用的服務的 API 金鑰及相關配置：

```env
# OpenAI API 設定
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Google API 設定 (Gemini)
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxx

# Anthropic API 設定 (Claude)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# DeepSeek API 設定
DEEPSEEK_API_KEY=xxxxxxxxxxxxxxxx
DEEPSEEK_API_URL=https://api.deepseek.com
DEEPSEEK_API_ENDPOINT_NONSTREAM=https://api.deepseek.com/chat/completions

# MinMax API 設定
MINMAX_API_KEY=xxxxxxxxxxxxxxxx
MINMAX_API_URL=https://api.minimax.chat/v1/text/chatcompletion_v2

# Ollama 服務設定 (本地或遠端)
OLLAMA_URL=http://localhost:11434
OLLAMA_API_KEY= # 如果您的 Ollama 服務需要 API 金鑰
OLLAMA_TIMEOUT=120 # 秒

# API 金鑰驗證設定（可選）
ENABLE_CHECK_APIKEY=False
api_keys_whitelist=
```

**注意事項**：
- 您只需要設定您計劃使用的服務的 API 金鑰。
- `DEEPSEEK_API_URL` 主要用於串流，`DEEPSEEK_API_ENDPOINT_NONSTREAM` 用於非串流。
- `OLLAMA_URL` 請對應您的本地或遠端 Ollama 服務位址。
- 若啟用 API 金鑰驗證，請將 `ENABLE_CHECK_APIKEY` 設為 `True`，並於 `api_keys_whitelist` 中填入允許的金鑰（多組以逗號分隔）。
- 請務必保護好您的 API 金鑰，不要將包含真實金鑰的 `.env` 檔案提交到公開的版本控制系統。

### 2.4 啟動伺服器
在您的專案目錄中執行：
```bash
python main.py
```
伺服器預設會在 `http://localhost:8000` 啟動。

或者，推薦使用 `uvicorn` 進行更專業的部署：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 3. 使用說明

### 3.1 文本嵌入 (Embedding)

#### 3.1.1 CURL 範例
**注意：以下 `Authorization: Bearer ...` 為本 API 伺服器驗證用金鑰，僅當本伺服器啟用 API 金鑰驗證功能時才需提供，請使用您於本系統產生的有效 API 金鑰。**

**OpenAI Embedding**：
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{
       "model": "text-embedding-3-small",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Google Gemini Embedding**：
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GOOGLE_API_KEY" \
     -d '{
       "model": "models/embedding-001",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Ollama Embedding**：
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OLLAMA_API_KEY" \
     -d '{
       "model": "ollama/nomic-embed-text",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Hugging Face Embedding**：
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $HF_TOKEN" \
     -d '{
       "model": "huggingface/bge-large-zh-v1.5",
       "input": "這是一段需要嵌入的文本。"
     }'
```

#### 3.1.2 Python (使用 `openai` 套件)
您可以使用官方的 `openai` Python 套件與此伺服器互動。

```python
from openai import OpenAI

# 設定您的伺服器位址
# 如果您的 main.py 伺服器本身不需要對傳入請求進行 API 金鑰驗證，
# api_key 可以設置為一個任意的非空字串。
client = OpenAI(
    base_url="http://localhost:8000/v1", # 指向您 main.py 伺服器的位址
    api_key="dummy-key" # 此金鑰不會被 main.py 驗證，可為任意值
)

try:
    embedding_response = client.embeddings.create(
        model="huggingface/sentence-transformers/all-MiniLM-L6-v2", # 或其他支援的 embedding 模型
        input="我喜歡在陽光明媚的日子裡散步。"
    )
    if not embedding_response.data:
        print("API 回應中沒有 embedding data。")
    else:
        print("Embedding 向量 (前5個維度):")
        print(embedding_response.data[0].embedding[:5])
        print(f"模型: {embedding_response.model}")
        print(f"總 Token 用量: {embedding_response.usage.total_tokens}")

except Exception as e:
    print(f"呼叫 Embedding API 時發生錯誤: {e}")
```

### 3.2 聊天補全 (Chat Completion)

#### 3.2.1 CURL 範例
**注意：以下 `Authorization: Bearer ...` 為本 API 伺服器驗證用金鑰，僅當本伺服器啟用 API 金鑰驗證功能時才需提供，請使用您於本系統產生的有效 API 金鑰。**

**OpenAI GPT (非串流)**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{
       "model": "openai/gpt-3.5-turbo",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**OpenAI GPT (串流)**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{
       "model": "openai/gpt-3.5-turbo",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7,
       "stream": true
     }'
```

**Anthropic Claude (非串流)**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     -d '{
       "model": "claude/claude-3-haiku-20240307",
       "messages": [
         {"role": "system", "content": "你是一個翻譯專家，請將以下英文翻譯成繁體中文。"},
         {"role": "user", "content": "Hello, how are you today?"}
       ],
       "temperature": 0.5,
       "max_tokens": 100
     }'
```

**Google Gemini (串流)**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GOOGLE_API_KEY" \
     -d '{
       "model": "gemini/gemini-1.5-flash-latest",
       "messages": [
         {"role": "system", "content": "你是一位詩人。"},
         {"role": "user", "content": "寫一首關於春天的小詩。"}
       ],
       "temperature": 0.8,
       "stream": true
     }'
```

#### 3.2.2 Python (使用 `openai` 套件)
**非串流範例**：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

try:
    chat_completion = client.chat.completions.create(
        model="deepseek/deepseek-chat",  # 替換成您想使用的模型
        messages=[
            {"role": "system", "content": "你是一個樂於助人的AI助理。"},
            {"role": "user", "content": "你好，請用繁體中文介紹一下你自己。"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    if chat_completion.choices:
        print("AI 回應:")
        print(chat_completion.choices[0].message.content)
        print(f"\n模型: {chat_completion.model}")
        print(f"ID: {chat_completion.id}")
        print(f"Token 用量: {chat_completion.usage}")
    else:
        print("API 回應中沒有 choices。")

except Exception as e:
    print(f"呼叫 Chat Completion API 時發生錯誤: {e}")
```

**串流範例**：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

try:
    print("AI 串流回應:")
    stream = client.chat.completions.create(
        model="openai/gpt-3.5-turbo", # 替換成您想使用的模型
        messages=[
            {"role": "user", "content": "寫一個關於太空探險的短故事。"}
        ],
        stream=True,
        temperature=0.8
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n\n串流結束。")

except Exception as e:
    print(f"呼叫 Chat Completion API (串流) 時發生錯誤: {e}")
```

### 3.3 API 金鑰管理與監控
本服務支援 API 金鑰驗證與管理，增強安全性與可追蹤性。

#### 3.3.1 啟用 API 金鑰驗證
- 在 `.env` 檔案中設定：
  ```env
  ENABLE_CHECK_APIKEY=True
  ```
- 並於 `api_keys_whitelist` 變數中設定允許的 API 金鑰（多組以逗號分隔）。
- 若啟用驗證但白名單為空，所有請求將被拒絕。

#### 3.3.2 API 金鑰管理介面
- 建議使用 `demo_web/admin.py` Streamlit 管理介面：
  - 產生新金鑰（自動加入白名單並更新 .env）
  - 啟用/停用/刪除金鑰
  - 編輯金鑰描述
- **請勿手動編輯 .env 的 api_keys_whitelist，避免設定不一致。**

#### 3.3.3 API 金鑰使用記錄
- 啟用驗證後，所有 API 請求會自動記錄於：
  - `history_apikey/{API_KEY}/{YYYYMMDD}.txt`
- 每筆記錄包含：
  - 請求時間、請求類型、模型、API 金鑰後綴、token 使用量、輸入/輸出摘要、狀態碼等

### 3.4 聊天示範頁面
本專案提供一個聊天頁面，示範如何應用 Open LLM Gateway 進行即時對話。您可以透過此頁面測試不同模型的聊天功能，體驗串流回應的即時性。

## 4. 範例腳本：`openai_client_example.py`

為了幫助您快速上手，我們提供了一個範例腳本 `openai_client_example.py`，展示了如何使用 OpenAI Python 函式庫與 Open LLM Gateway 互動。該腳本包含以下功能：

- **非串流聊天補全**：一次性接收模型的完整回應。
- **串流聊天補全**：即時接收模型回應的小塊內容，適合長篇回應或即時互動。
- **文本嵌入**：將文本轉換為向量表示，用於語義搜索或相似性比較等任務。

### 4.1 使用步驟
1. 確保您的 Open LLM Gateway 伺服器正在運行。
2. 安裝必要的 Python 函式庫：
   ```bash
   pip install openai
   ```
3. 修改 `openai_client_example.py` 中的組態設定：
   - 設定 `OPENLLM_GATEWAY_BASE_URL` 為您的 Gateway API 端點，例如 `http://localhost:8000/v1`。
   - 設定 `OPENLLM_GATEWAY_API_KEY` 為您的 API 金鑰（若 Gateway 未啟用金鑰檢查，可設為任意非空字串）。
   - 選擇您要使用的聊天模型 (`CHAT_MODEL_NAME`) 和嵌入模型 (`EMBEDDING_MODEL_NAME`)。
4. 執行腳本：
   ```bash
   python openai_client_example.py
   ```

腳本將依序執行非串流聊天、串流聊天和文本嵌入的範例，並顯示結果。透過此範例，您可以快速了解如何將 Open LLM Gateway 整合到自己的專案中。

## 5. 注意事項
- **API 金鑰安全**：切勿將您的 API 金鑰直接硬編碼到程式中或提交到公開的程式碼庫。優先使用 `.env` 檔案管理。
- **錯誤處理**：不同 LLM 供應商的 API 可能返回不同格式的錯誤訊息。用戶端應實作適當的錯誤處理邏輯。
- **模型名稱**：調用 API 時，請確保使用的 `model` 名稱對於所選供應商是有效的。
- **速率限制**：各雲端 LLM 服務均有其 API 呼叫速率限制。高頻率使用時請查閱對應供應商的文檔，並在客戶端實施適當的重試或節流機制。
- **服務可用性**：若某一供應商服務暫時不可用，此閘道設計允許您相對輕鬆地切換到其他可用的模型供應商。 