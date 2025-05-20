'''
Open LLM Gateway - OpenAI Client Python Example

本腳本旨在演示如何使用 OpenAI Python 函式庫，透過 Open LLM Gateway 進行以下操作：
- 非串流 (Non-streaming) 聊天補全
- 串流 (Streaming) 聊天補全
- 文本嵌入 (Text Embeddings)

執行本腳本前的準備工作：
1. 您的 Open LLM Gateway 伺服器 (通常是 main.py) 必須正在運行。
2. 確認已安裝 `openai` Python 函式庫 (若無，請執行 `pip install openai`)。
3. 在下方「組態設定」區塊中，正確設定 `OPENLLM_GATEWAY_BASE_URL` 和 `OPENLLM_GATEWAY_API_KEY`。
   - `OPENLLM_GATEWAY_BASE_URL`：應指向您 Gateway 的 `/v1` API 端點。
   - `OPENLLM_GATEWAY_API_KEY`：
     - 如果您的 Gateway 伺服器啟用金鑰檢查 (main.py 中的 `ENABLE_CHECK_APIKEY=True`)，
       此處必須填寫一個已在伺服器 `api_keys_whitelist` 中設定的有效金鑰。
     - 如果您的 Gateway 伺服器禁用金鑰檢查 (`ENABLE_CHECK_APIKEY=False`)，
       此處可以填寫任意非空字串 (例如："not_needed")。
4. 確認在範例中所使用的模型名稱 (例如：`CHAT_MODEL_NAME`, `EMBEDDING_MODEL_NAME`)
   已在您的 Open LLM Gateway 伺服器上正確設定並支援。
   模型名稱的格式通常是 `provider/model_identifier`，例如：
   - Ollama: "ollama/mistral", "ollama/qwen2:7b"
   - OpenAI: "openai/gpt-3.5-turbo", "openai/gpt-4"
   - Gemini: "gemini/gemini-pro", "gemini/gemini-1.5-flash"
   - HuggingFace (用於 Embedding): "huggingface/sentence-transformers/all-MiniLM-L6-v2"
   - HuggingFace (用於 Chat, 若伺服器有此實驗性支援): "huggingface/microsoft/Phi-3-mini-4k-instruct"
'''
import os
from openai import OpenAI
from openai import APIError # 引入 APIError 以便更精確地捕捉錯誤

# --- 組態設定 ---
# 請根據您的 Open LLM Gateway 實際運行情況修改以下變數

# 您的 Open LLM Gateway 的 `/v1` API 端點 URL
# 例如：如果您在本機運行 Gateway 且端口為 8000，則通常是 "http://localhost:8000/v1"
OPENLLM_GATEWAY_BASE_URL = "http://localhost:8000/v1"

# 您的 API 金鑰
# - 若 Gateway 啟用金鑰檢查，請填寫白名單中的金鑰。
# - 若 Gateway 未啟用金鑰檢查，可填寫任意非空字串 (例如："not_needed")。
OPENLLM_GATEWAY_API_KEY = "DEFAULT-20250508-kbxk8c"  # <--- 請務必替換成您的 API 金鑰或適用字串

# 範例中使用的聊天模型名稱
# 請確保此模型已在您的 Gateway 中設定並可用。
CHAT_MODEL_NAME = "ollama/qwen3:8b"
# CHAT_MODEL_NAME = "openai/gpt-4.1"
# CHAT_MODEL_NAME = "gemini/gemini-2.0-flash"
# CHAT_MODEL_NAME = "claude/claude-3-7-sonnet-latest"

# 範例中使用的文本嵌入模型名稱
# 請確保此模型已在您的 Gateway 中設定並可用。
EMBEDDING_MODEL_NAME = "ollama/bge-m3:latest"
# EMBEDDING_MODEL_NAME = "openai/text-embedding-3-small"
# EMBEDDING_MODEL_NAME = "gemini/embedding-001"

# --- 初始化 OpenAI 客戶端 ---
# 設定 base_url 以指向您的 Open LLM Gateway，並提供 API 金鑰。
try:
    client = OpenAI(
        base_url=OPENLLM_GATEWAY_BASE_URL,
        api_key=OPENLLM_GATEWAY_API_KEY,
    )
    print(f"OpenAI 客戶端已成功初始化，指向 Gateway: {OPENLLM_GATEWAY_BASE_URL}")
except Exception as e:
    print(f"初始化 OpenAI 客戶端時發生嚴重錯誤: {e}")
    print("請檢查 OPENLLM_GATEWAY_BASE_URL 和 OPENLLM_GATEWAY_API_KEY 設定是否正確。")
    exit(1)
print("-" * 50)

def example_chat_completion_non_streaming():
    '''範例：非串流 (Non-Streaming) 聊天補全

    此範例演示如何向 Gateway 發送一個聊天請求，並一次性接收完整的模型回應。'''
    print(f"--- 範例：非串流聊天補全 (模型: {CHAT_MODEL_NAME}) ---")
    try:
        print("正在傳送請求至 Gateway...")
        completion = client.chat.completions.create(
            model=CHAT_MODEL_NAME,      # 指定要使用的模型
            messages=[                  # 對話歷史，包含系統訊息和使用者提問
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": "Hello! Can you tell me a short joke?"}
            ],
            temperature=0.7,            # 控制回應的隨機性，0 表示最確定，越高越隨機
            max_tokens=150              # 限制模型產生的最大 token 數量
        )
        print("\nGateway 回應成功!")
        print("Assistant 回答:", completion.choices[0].message.content)

        # `usage` 欄位包含了 token 使用量的資訊
        # 注意：並非所有模型或 Gateway 的實作都會完整提供此資訊。
        # Gemini 模型透過此 Gateway 的非串流 token 計數是基於字元長度估算。
        if completion.usage:
            print("Token 使用量 (由 Gateway 提供):")
            print(f"  - 提示 (Prompt) Tokens: {completion.usage.prompt_tokens}")
            print(f"  - 補全 (Completion) Tokens: {completion.usage.completion_tokens}")
            print(f"  - 總計 (Total) Tokens: {completion.usage.total_tokens}")
        else:
            print("Token 使用量資訊未提供。")

    except APIError as e:
        print(f"非串流聊天補全時發生 API 錯誤 (狀態碼: {e.status_code}):")
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                print(f"  錯誤詳情 (來自 Gateway): {error_detail.get('detail', e.message)}")
            except ValueError: # Non-JSON response
                 print(f"  錯誤訊息: {e.message} (回應非 JSON 格式)")
        else:
            print(f"  錯誤訊息: {e}")
    except Exception as e:
        print(f"非串流聊天補全時發生未預期錯誤: {e}")
    print("-" * 50 + "\n")

def example_chat_completion_streaming():
    '''範例：串流 (Streaming) 聊天補全

    此範例演示如何以串流方式接收模型的回應。
    Gateway 會將模型產生的內容分成小塊 (chunks) 即時回傳。'''
    print(f"--- 範例：串流聊天補全 (模型: {CHAT_MODEL_NAME}) ---")
    try:
        print("正在傳送串流請求至 Gateway...")
        stream = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in crafting short verses."},
                {"role": "user", "content": "Compose a brief, two-stanza poem about a silent, moonlit night."}
            ],
            temperature=0.8,
            stream=True                 # 關鍵參數：啟用串流模式
        )

        print("\nAssistant (串流回應中):")
        full_response_content = ""
        for chunk in stream:
            # 每個 chunk 包含回應的一小部分
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                print(content_piece, end="", flush=True) # 即時印出，不換行
                full_response_content += content_piece
        
        print("\n\n--- 串流結束 ---")
        print(f"收集到的完整串流回應內容: \n{full_response_content}")

        # 關於串流模式下的 token 使用量:
        # - 標準 OpenAI API 在串流模式下，最後一個 chunk 可能包含 `usage` 資訊 (取決於庫版本和模型)。
        # - Open LLM Gateway 的日誌系統 (log_api_usage) 會在串流結束後，
        #   根據收集到的完整提示和回應內容來記錄估算的 token 數量。
        # - 此範例客戶端主要演示接收串流內容，不直接處理來自串流的 `usage` 物件。
        print("注意：串流模式的 token 使用量主要由 Gateway 伺服器在其日誌中估算和記錄。")

    except APIError as e:
        print(f"\n串流聊天補全時發生 API 錯誤 (狀態碼: {e.status_code}):")
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                print(f"  錯誤詳情 (來自 Gateway): {error_detail.get('detail', e.message)}")
            except ValueError:
                 print(f"  錯誤訊息: {e.message} (回應非 JSON 格式)")
        else:
            print(f"  錯誤訊息: {e}")
    except Exception as e:
        print(f"\n串流聊天補全時發生未預期錯誤: {e}")
    print("-" * 50 + "\n")

def example_embedding():
    '''範例：文本嵌入 (Text Embeddings)

    此範例演示如何將文本轉換為向量表示 (embeddings)。'''
    print(f"--- 範例：文本嵌入 (模型: {EMBEDDING_MODEL_NAME}) ---")
    texts_to_embed = [
        "Open LLM Gateway 是一個強大的工具。",
        "文本嵌入可用於語義搜索、相似性比較等任務。"
    ]
    print(f"待嵌入的文本: {texts_to_embed}")

    try:
        print("正在向 Gateway 請求文本嵌入...")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, # 指定嵌入模型
            input=texts_to_embed        # 要嵌入的文本列表 (或單個字串)
        )
        print("\nGateway 回應成功!")
        for i, data_item in enumerate(response.data):
            # data_item 是 EmbeddingData 類型的物件
            print(f"文本 {i+1} 的嵌入向量 (前3個維度): {data_item.embedding[:3]}...")
            print(f"  - 向量維度: {len(data_item.embedding)}")
            print(f"  - 物件類型: {data_item.object}")
            print(f"  - 索引: {data_item.index}")

        if response.usage:
            print("Token 使用量 (由 Gateway 提供):")
            print(f"  - 提示 (Prompt) Tokens: {response.usage.prompt_tokens}")
            # 嵌入操作通常只有 prompt_tokens，completion_tokens 為 0
            print(f"  - 總計 (Total) Tokens: {response.usage.total_tokens}")
        else:
            print("Token 使用量資訊未提供。")
        
        print(f"回應中使用的模型: {response.model}")


    except APIError as e:
        print(f"文本嵌入時發生 API 錯誤 (狀態碼: {e.status_code}):")
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                print(f"  錯誤詳情 (來自 Gateway): {error_detail.get('detail', e.message)}")
            except ValueError:
                 print(f"  錯誤訊息: {e.message} (回應非 JSON 格式)")
        else:
            print(f"  錯誤訊息: {e}")
    except Exception as e:
        print(f"文本嵌入時發生未預期錯誤: {e}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    print("Open LLM Gateway - OpenAI 客戶端範例腳本")
    print("=" * 50)

    if OPENLLM_GATEWAY_API_KEY == "DEFAULT-20250508-kbxk8c": # 這裡用您實際填寫的預設金鑰或提示替換
        print("提示：目前使用的是預設的 API 金鑰。")
        print("請確保此金鑰對於您的 Gateway 設定是有效的，或者根據您的 Gateway 安全設定進行相應的修改。")
        print("如果您的 Gateway 未啟用金鑰檢查，目前的設定應該可以運作。")
        print("-" * 50)
    elif not OPENLLM_GATEWAY_API_KEY:
        print("錯誤：OPENLLM_GATEWAY_API_KEY 未設定！")
        print("請在本腳本開頭的「組態設定」區塊中設定您的 API 金鑰。")
        print("如果您的 Gateway 未啟用金鑰檢查，可以將其設定為任意非空字串。")
        exit(1)


    # 執行所有範例函式
    example_chat_completion_non_streaming()
    example_chat_completion_streaming()
    example_embedding()

    print("所有範例已執行完畢。")
    print("=" * 50) 
