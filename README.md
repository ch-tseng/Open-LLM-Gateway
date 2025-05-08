# OpenAI 相容 API 伺服器 (多模型支援)

這是一個使用 FastAPI 建置的 Python 伺服器，提供與 OpenAI API 相容的介面，支援多種 AI 服務。

## 1. 功能說明

### 1.1 主要特性
*   **API 相容性**: 與 OpenAI API 格式高度相容，方便現有專案無縫遷移。
*   **多模型支援**: 整合多種主流 LLM 供應商及本地模型，只需修改模型名稱即可切換。
*   **Embedding**: 提供高效的文本嵌入功能，支援多種來源。
*   **Chat Completion**: 提供強大的聊天對話功能，支援串流與非串流模式。
*   **串流輸出 (Streaming)**: 所有聊天模型供應商均支援串流回應，可即時獲取生成內容。

### 1.2 Embedding 功能支援（使用者可自行指定）
*   **OpenAI API**: 例如 `text-embedding-3-small`, `text-embedding-ada-002`
*   **Google Gemini API**: 例如 `models/embedding-001`
*   **Ollama**: 本地運行的 Embedding 模型，例如 `ollama/nomic-embed-text`
*   **Hugging Face 模型**: 使用 `sentence-transformers` 載入模型，例如 `huggingface/bge-large-zh-v1.5` 或直接使用 `sentence-transformers/all-MiniLM-L6-v2` (向後兼容)

### 1.3 Chat Completion 功能支援（使用者可自行指定）
*   **OpenAI API**: 例如 `openai/gpt-4`, `openai/gpt-3.5-turbo`
*   **Anthropic Claude API**: 例如 `claude/claude-3-opus-20240229`, `claude/claude-3-sonnet-20240229`, `claude/claude-3-haiku-20240307`
*   **Google Gemini API**: 例如 `gemini/gemini-1.5-pro-latest`, `gemini/gemini-1.0-pro`
*   **DeepSeek API**: 例如 `deepseek/deepseek-chat`
*   **MinMax API**: 例如 `minmax/abab6-chat`
*   **Ollama**: 本地運行的聊天模型，例如 `ollama/llama3`, `ollama/mistral`
*   **Hugging Face 模型**: 本地運行的開源聊天模型，例如 `huggingface/meta-llama/Llama-3-8B-Instruct`, `huggingface/mistralai/Mistral-7B-Instruct-v0.2` (需注意本地資源需求)

## 2. 安裝及設定

### 2.1 環境需求
*   Python 3.8 或更高版本
*   `pip` 套件管理器
*   網際網路連線 (用於下載模型或連接雲端 API 服務)

### 2.2 安裝步驟
1.  **複製專案**:
    將 `main.py`, `requirements.txt`, `README.md` 和 (可選的) `.env.example` (需自行複製為 `.env`) 檔案放在您的專案目錄中。

2.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```

### 2.3 API 金鑰設定 (重要!)
將 `.env.example` 檔案複製一份並重新命名為 `.env`。然後，在 `.env` 檔案中設定您需要使用的服務的 API 金鑰及相關配置：

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
OLLAMA_BASEURL=http://localhost:11434
OLLAMA_API_KEY= # 如果您的 Ollama 服務需要 API 金鑰
OLLAMA_TIMEOUT=120 # 秒

# HuggingFace Token (可選，用於訪問私有模型或某些需要登入的模型)
# HF_TOKEN=your_huggingface_token
```

**注意事項**:
*   您只需要設定您計劃使用的服務的 API 金鑰。
*   `DEEPSEEK_API_URL` 主要用於串流，`DEEPSEEK_API_ENDPOINT_NONSTREAM` 用於非串流。
*   請務必保護好您的 API 金鑰，不要將包含真實金鑰的 `.env` 檔案提交到公開的版本控制系統。

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

## 3. Embedding 的使用

### 3.1 CURL 範例

**OpenAI Embedding**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "text-embedding-3-small",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Google Gemini Embedding**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "models/embedding-001",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Ollama Embedding**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "ollama/nomic-embed-text",
       "input": "這是一段需要嵌入的文本。"
     }'
```

**Hugging Face Embedding**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "huggingface/bge-large-zh-v1.5",
       "input": "這是一段需要嵌入的文本。"
     }'
```

### 3.2 Python (使用 `openai` 套件)
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

### 3.3 Python (使用 `requests` 套件)

```python
import requests
import json

# 設定 API 端點
api_url = "http://localhost:8000/v1/embeddings"
headers = {"Content-Type": "application/json"}

# 選擇模型
model_name = "text-embedding-3-small"       # OpenAI
# model_name = "models/embedding-001"       # Google Gemini
# model_name = "ollama/nomic-embed-text"    # Ollama
# model_name = "huggingface/bge-large-zh-v1.5"  # Hugging Face

# 準備請求數據
data = {
    "model": model_name,
    "input": "這是一段需要嵌入的文本。"
}

# 發送請求
try:
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # 檢查是否有錯誤
    
    result = response.json()
    print(f"模型: {result['model']}")
    print(f"向量維度: {len(result['data'][0]['embedding'])}")
    # print(f"使用的 Tokens: {result['usage']['total_tokens']}") # 注意：HuggingFace 本地模型可能無法精確回報 token
    
    embeddings = result['data'][0]['embedding']
    print(f"嵌入向量 (前5個值): {embeddings[:5]}")
    
except requests.exceptions.RequestException as e:
    print(f"請求錯誤: {e}")
    if hasattr(e, 'response') and e.response:
        try:
            error_detail = e.response.json()
            print(f"錯誤詳情: {error_detail}")
        except:
            print(f"錯誤狀態碼: {e.response.status_code}")
            print(f"錯誤內容: {e.response.text}")
```

## 4. Chat Completion 的使用

### 4.1 CURL 範例

**OpenAI GPT (非串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "openai/gpt-3.5-turbo",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**OpenAI GPT (串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
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
**Anthropic Claude (非串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
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

**Google Gemini (串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
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

**DeepSeek (非串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "deepseek/deepseek-chat",
       "messages": [
         {"role": "user", "content": "解釋什麼是量子糾纏，用簡單易懂的方式。"}
       ],
       "temperature": 0.7
     }'
```

**MinMax (串流)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "minmax/abab6-chat",
       "messages": [
         {"role": "system", "content": "你是一個歷史學家。"},
         {"role": "user", "content": "簡述一下羅馬帝國的崛起與衰落。"}
       ],
       "temperature": 0.6,
       "stream": true
     }'
```

**Ollama (非串流，使用 llama3)**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "ollama/llama3",
       "messages": [
         {"role": "user", "content": "你好！"}
       ],
       "temperature": 0.7
     }'
```

**Hugging Face (非串流，使用 Llama-3-8B-Instruct)**:
```bash
# 注意: 首次運行 Hugging Face 模型可能需要較長時間下載模型檔案
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "huggingface/meta-llama/Llama-3-8B-Instruct",
       "messages": [
         {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
         {"role": "user", "content": "What is the capital of France?"}
       ],
       "temperature": 0.2,
       "max_tokens": 50
     }'
```

### 4.2 Python (使用 `openai` 套件)

**非串流範例**:
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

**串流範例**:
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

### 4.3 Python (使用 `requests` 套件)

**非串流範例**:
```python
import requests
import json

api_url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

model_name = "openai/gpt-3.5-turbo"
# model_name = "claude/claude-3-haiku-20240307"
# model_name = "gemini/gemini-1.5-flash-latest"
# model_name = "deepseek/deepseek-chat"
# model_name = "minmax/abab6-chat"
# model_name = "ollama/llama3"
# model_name = "huggingface/meta-llama/Llama-3-8B-Instruct"

data = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "你是一個有用的助手。"},
        {"role": "user", "content": "請介紹臺灣的夜市文化。"}
    ],
    "temperature": 0.7
}

try:
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    
    result = response.json()
    if result.get('choices') and len(result['choices']) > 0:
        message = result['choices'][0]['message']['content']
        print(f"模型回應:\n{message}")
        if 'usage' in result:
            print(f"\nTokens: {result['usage']}")
    else:
        print("未收到有效回應內容")
    
except requests.exceptions.RequestException as e:
    print(f"請求錯誤: {e}")
    if hasattr(e, 'response') and e.response:
        try:
            print(f"錯誤詳情: {e.response.json()}")
        except:
            print(f"錯誤內容: {e.response.text}")
```

**串流範例**:
```python
import requests
import json
# 串流處理可能需要 sseclient-py: pip install sseclient-py
# from sseclient import SSEClient # 或者手動解析串流

api_url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

model_name = "openai/gpt-3.5-turbo"
# ... (其他模型選擇同上) ...

data = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "請用中文介紹臺灣的夜市文化，並列出5個著名夜市。"}
    ],
    "temperature": 0.7,
    "stream": True
}

try:
    with requests.post(api_url, headers=headers, data=json.dumps(data), stream=True) as response:
        response.raise_for_status()
        print("模型串流回應:\n")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_data = decoded_line[len('data: '):]
                    if json_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_data)
                        if chunk.get('choices') and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        # print(f"無法解析JSON: {json_data}")
                        continue
        print("\n\n串流回應結束")
except requests.exceptions.RequestException as e:
    print(f"請求錯誤: {e}")
    # ... (錯誤處理同上) ...
```

## 5. Hugging Face 模型額外說明
*   **資源需求**: 本地運行 Hugging Face 的聊天模型 (尤其是大型模型) 對硬體資源 (如 GPU 記憶體、CPU、磁碟空間) 有較高要求。
*   **首次下載**: 首次使用特定 Hugging Face 模型時，系統會自動從 Hugging Face Hub 下載模型權重，可能需要較長時間。
*   **模型兼容性**:
    *   Embedding: 主要支援 `sentence-transformers` 相容模型。
    *   Chat Completion: 主要支援可用於 `AutoModelForCausalLM` 的模型，並建議使用針對指令微調 (instruct-tuned) 或聊天優化 (chat-tuned) 的版本以獲得更佳效果。
*   **Token計算**: 對於本地 Hugging Face 模型，`usage`中的 token 計數可能是估算值或未提供。

## 6. 注意事項
*   **API 金鑰安全**: 再次強調，切勿將您的 API 金鑰直接硬編碼到程式中或提交到公開的程式碼庫。優先使用 `.env` 檔案管理。
*   **錯誤處理**: 不同 LLM 供應商的 API 可能返回不同格式的錯誤訊息。用戶端應實作適當的錯誤處理邏輯。
*   **模型名稱**: 調用 API 時，請確保使用的 `model` 名稱對於所選供應商是有效的。
*   **速率限制**: 各雲端 LLM 服務均有其 API 呼叫速率限制。高頻率使用時請查閱對應供應商的文檔，並在客戶端實施適當的重試或節流機制。
*   **服務可用性**: 若某一供應商服務暫時不可用，此閘道設計允許您相對輕鬆地切換到其他可用的模型供應商。 
