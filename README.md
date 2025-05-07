# OpenAI 相容 API 伺服器 (多模型支援)

這是一個使用 FastAPI 建置的 Python 伺服器，提供與 OpenAI API 相容的介面，支援多種 AI 服務。

## 1. 功能介紹

### 1.1 Embedding 功能

支援以下 Embedding 來源：
*   **OpenAI API**: 呼叫 OpenAI 的 Embedding API (例如 `text-embedding-3-small`, `text-embedding-ada-002`)
*   **Google Gemini API**: 呼叫 Google Generative AI 的 Embedding API (例如 `models/embedding-001`)
*   **Ollama**: 透過 Ollama 服務呼叫本地運行的 Embedding 模型 (例如 `ollama/nomic-embed-text`)
*   **本地 Hugging Face 模型**: 使用 `sentence-transformers` 函式庫動態載入 Hugging Face Hub 上的任何相容模型

### 1.2 Chat Completion 功能

支援以下聊天模型服務：
*   **OpenAI API**: 支援 GPT 模型 (例如 `openai/gpt-4`, `openai/gpt-3.5-turbo`)
*   **Anthropic Claude**: 支援 Claude 模型 (例如 `claude/claude-3-haiku-20240307`, `claude/claude-3-sonnet-20240229`)
*   **Google Gemini**: 支援 Gemini 模型 (例如 `gemini/gemini-1.5-pro`, `gemini/gemini-1.0-pro`)
*   **DeepSeek**: 支援 DeepSeek 聊天模型 (例如 `deepseek/deepseek-chat`)
*   **MinMax**: 支援 MinMax 聊天模型 (例如 `minmax/abab6-chat`)
*   **Ollama**: 支援本地部署的開源模型 (例如 `ollama/llama3`, `ollama/qwen:7b`)

### 1.3 特殊功能

* **串流輸出 (Streaming)**: 所有聊天模型皆支援串流回應，即時返回生成內容
* **模型切換**: 使用相同的程式碼，只需更改模型名稱前綴即可切換不同服務
* **API 相容性**: 與 OpenAI API 格式相容，原使用 OpenAI 的專案可輕鬆整合

## 2. 安裝和設定

### 2.1 環境需求

* Python 3.8 或更高版本
* 網際網路連線 (用於連接雲端 API 服務)

### 2.2 安裝步驟

1. **複製專案**:
   將 `main.py`, `requirements.txt`, `README.md` 和 (可選) `.env` 檔案放在專案目錄中。

2. **安裝依賴**:
   ```bash
   pip install -r requirements.txt
   ```

### 2.3 API 金鑰設定 (重要!)

建立 `.env` 檔案，並設定以下環境變數：

```bash
# OpenAI API 設定
openai_api=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Google API 設定
gemini_api=AIzaSyxxxxxxxxxxxxxxxxxx

# Anthropic API 設定
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# DeepSeek API 設定
deepseek_api=xxxxxxxxxxxxxxxx
deepseek_api_url=https://api.deepseek.com

# MinMax API 設定
minmax_api=xxxxxxxxxxxxxxxx
minmax_baseurl=https://api.minimax.chat/v1/text/chatcompletion_v2

# Ollama 服務設定 (本地或遠端)
ollama_baseurl=http://localhost:11434
ollama_timeout=120
```

**注意事項**：
* 您只需設定您計劃使用的服務的金鑰
* 請確保 API 金鑰的安全，不要將包含實際金鑰的 `.env` 檔案提交到版本控制系統

### 2.4 啟動伺服器

```bash
python main.py
```

或使用 uvicorn (建議用於生產環境)：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 3. CURL 使用範例

### 3.1 Embedding 使用範例

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
       "model": "all-MiniLM-L6-v2",
       "input": "這是一段需要嵌入的文本。"
     }'
```

### 3.2 Chat Completion 使用範例

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

**Anthropic Claude**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "claude/claude-3-haiku-20240307",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**Google Gemini**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gemini/gemini-1.5-flash",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**DeepSeek**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "deepseek/deepseek-chat",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**MinMax**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "minmax/abab6-chat",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

**Ollama**:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "ollama/llama3",
       "messages": [
         {"role": "system", "content": "你是一個有用的助手。"},
         {"role": "user", "content": "請介紹臺灣的夜市文化。"}
       ],
       "temperature": 0.7
     }'
```

## 4. Python 使用範例

### 4.1 Embedding 使用範例

```python
import requests
import json
import os

# 設定 API 端點
api_url = "http://localhost:8000/v1/embeddings"
headers = {"Content-Type": "application/json"}

# 選擇模型
model_name = "text-embedding-3-small"       # OpenAI
# model_name = "models/embedding-001"       # Google Gemini
# model_name = "ollama/nomic-embed-text"    # Ollama
# model_name = "all-MiniLM-L6-v2"           # Hugging Face

# 準備請求數據
data = {
    "model": model_name,
    "input": "這是一段需要嵌入的文本。"
}

# 發送請求
try:
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # 檢查是否有錯誤
    
    # 解析回應
    result = response.json()
    print(f"模型: {result['model']}")
    print(f"向量維度: {len(result['data'][0]['embedding'])}")
    print(f"使用的 Tokens: {result['usage']['total_tokens']}")
    
    # 顯示部分嵌入向量
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

### 4.2 Chat Completion 使用範例 (非串流)

```python
import requests
import json
import os

# 設定 API 端點
api_url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 選擇模型
model_name = "openai/gpt-3.5-turbo"         # OpenAI
# model_name = "claude/claude-3-haiku-20240307"  # Anthropic Claude
# model_name = "gemini/gemini-1.5-flash"    # Google Gemini
# model_name = "deepseek/deepseek-chat"     # DeepSeek
# model_name = "minmax/abab6-chat"          # MinMax
# model_name = "ollama/llama3"              # Ollama

# 準備請求數據
data = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "你是一個有用的助手。"},
        {"role": "user", "content": "請介紹臺灣的夜市文化。"}
    ],
    "temperature": 0.7
}

# 發送請求
try:
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # 檢查是否有錯誤
    
    # 解析回應
    result = response.json()
    
    # 提取並顯示回應內容
    if result.get('choices') and len(result['choices']) > 0:
        message = result['choices'][0]['message']['content']
        print(f"模型回應:\n{message}")
        
        # 顯示使用的 tokens (如果有提供)
        if 'usage' in result:
            usage = result['usage']
            print(f"\n使用的 Tokens: 提示 {usage.get('prompt_tokens', 'N/A')}, " +
                  f"回應 {usage.get('completion_tokens', 'N/A')}, " +
                  f"總計 {usage.get('total_tokens', 'N/A')}")
    else:
        print("未收到有效回應內容")
    
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

### 4.3 Chat Completion 使用範例 (串流)

```python
import requests
import json
import os
import sseclient  # 需先安裝: pip install sseclient-py

# 設定 API 端點
api_url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 選擇模型
model_name = "openai/gpt-3.5-turbo"         # OpenAI
# model_name = "claude/claude-3-haiku-20240307"  # Anthropic Claude
# model_name = "gemini/gemini-1.5-flash"    # Google Gemini
# model_name = "deepseek/deepseek-chat"     # DeepSeek
# model_name = "minmax/abab6-chat"          # MinMax
# model_name = "ollama/llama3"              # Ollama

# 準備請求數據 (注意啟用串流)
data = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "你是一個有用的助手。"},
        {"role": "user", "content": "請用中文介紹臺灣的夜市文化，並列出5個著名夜市。"}
    ],
    "temperature": 0.7,
    "stream": True  # 啟用串流模式
}

# 發送請求並處理串流回應
try:
    with requests.post(api_url, headers=headers, data=json.dumps(data), stream=True) as response:
        response.raise_for_status()  # 檢查是否有錯誤
        
        print("模型回應:\n")
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break
            
            try:
                chunk = json.loads(event.data)
                if chunk.get('choices') and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                continue
            
        print("\n\n串流回應結束")
    
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

## 5. 串流功能說明

所有聊天模型均支援串流輸出 (Streaming)，適合需要即時顯示回應的場景。啟用方法為在請求中設定 `"stream": true`。

### 串流回應格式
```
data: {"choices": [{"delta": {"content": "部分回應內容"}}]}
data: {"choices": [{"delta": {"content": "更多回應內容"}}]}
data: [DONE]
```

## 6. 注意事項

* **API 金鑰安全**: 使用環境變數或 `.env` 檔案管理 API 金鑰，避免直接寫入程式碼
* **錯誤處理**: 各服務可能回傳不同的錯誤格式，建議實作穩健的錯誤處理機制
* **模型選擇**: 確保使用所選服務支援的模型名稱，否則會收到錯誤
* **速率限制**: 各服務可能有不同的速率限制，高流量應用應考慮客戶端的節流機制
* **服務可用性**: 若特定服務暫時不可用，可輕鬆切換到其他服務 