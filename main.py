from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any, Optional
import time
import os
from dotenv import load_dotenv # 可選，用於從 .env 檔案載入環境變數
import openai
from google import genai
import tiktoken
import ollama
from ollama import ResponseError as OllamaResponseError  # 正確導入 OllamaResponseError
import requests
import aiohttp
from fastapi.responses import StreamingResponse
import json
import asyncio

# --- Pydantic 模型定義 ---

class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str # 模型名稱，我們將用它來選擇或驗證
    # encoding_format: Optional[str] = "float" # 可以稍後添加支援
    # user: Optional[str] = None # 可以稍後添加支援

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str # 回傳使用的模型名稱
    usage: Usage

# 新增 Chat 相關模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# --- 環境變數與設定 (增加) ---
load_dotenv() # 可選

# 載入 .env 檔案（只需要一次）
load_dotenv()

# --- API 金鑰驗證相關 (新增) ---
ENABLE_CHECK_APIKEY = os.getenv("ENABLE_CHECK_APIKEY", "False").lower() == "true"
API_KEYS_WHITELIST_STR = os.getenv("api_keys_whitelist", "")
API_KEYS_WHITELIST = {key.strip() for key in API_KEYS_WHITELIST_STR.split(',') if key.strip()}

if ENABLE_CHECK_APIKEY:
    print("INFO: API 金鑰檢查已啟用。")
    if not API_KEYS_WHITELIST:
        print("警告：API 金鑰檢查已啟用，但 api_keys_whitelist 為空。所有需要金鑰的請求都可能被拒絕，除非白名單被正確設定。")
    else:
        # 為安全起見，不在日誌中打印整個白名單，只打印數量或部分信息
        print(f"INFO: 已載入 {len(API_KEYS_WHITELIST)} 個 API 金鑰到白名單。")
else:
    print("INFO: API 金鑰檢查已停用。")

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """
    驗證 API 金鑰。作為 FastAPI 依賴項使用。
    """
    if not ENABLE_CHECK_APIKEY:
        return # 如果未啟用檢查，則直接通過

    if not authorization:
        print("DEBUG: 驗證失敗 - 未提供 Authorization header。")
        raise HTTPException(
            status_code=401,
            detail="未提供 API 金鑰。請在 'Authorization' header 中提供金鑰 (格式: 'Bearer YOUR_API_KEY')。"
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        print(f"DEBUG: 驗證失敗 - Authorization header 格式錯誤: {authorization}")
        raise HTTPException(
            status_code=401,
            detail="API 金鑰格式錯誤。應為 'Bearer YOUR_API_KEY'。"
        )

    token = parts[1]
    if token not in API_KEYS_WHITELIST:
        # 為安全起見，不在日誌中記錄嘗試失敗的 token，或只記錄部分 hash
        print(f"DEBUG: 驗證失敗 - API 金鑰 '{token[:4]}...' 不在白名單中。")
        raise HTTPException(
            status_code=403,
            detail="提供的 API 金鑰無效或沒有權限。"
        )

    print(f"DEBUG: API 金鑰 '{token[:4]}...' 驗證成功。")
    # 此處不需要回傳值，如果沒有拋出 HTTPException，FastAPI 會認為依賴項已滿足
    return

# 讀取所有環境變數
OPENAI_API_KEY = os.getenv("openai_api")
GOOGLE_API_KEY = os.getenv("gemini_api")
OLLAMA_URL = os.getenv("ollama_baseurl")
OLLAMA_API_KEY = os.getenv("ollama_api")
OLLAMA_TIMEOUT = int(os.getenv("ollama_timeout", "120"))
DEEPSEEK_API_KEY = os.getenv("deepseek_api")
DEEPSEEK_API_URL = os.getenv("deepseek_api_url", "https://api.deepseek.com") # For streaming base
DEEPSEEK_API_ENDPOINT_NONSTREAM = os.getenv("DEEPSEEK_API_ENDPOINT_NONSTREAM", "https://api.deepseek.com/chat/completions") # For non-streaming full endpoint
MINMAX_API_KEY = os.getenv("minmax_api")
MINMAX_API_URL = os.getenv("minmax_baseurl", "https://api.minimax.chat/v1/text/chatcompletion_v2")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") # New key for Claude

# 設定 OpenAI 和 Google 的環境變數（如果需要）
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 使用環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_URL = os.getenv("ollama_baseurl")  # 直接使用 .env 中的完整 URL
OLLAMA_API_KEY = os.getenv("ollama_api")  # 從 .env 讀取 API 金鑰
OLLAMA_TIMEOUT = int(os.getenv("ollama_timeout", "120"))  # 預設 120 秒

# 設定 DeepSeek 的環境變數
DEEPSEEK_API_KEY = os.getenv("deepseek_api")
DEEPSEEK_API_URL = os.getenv("deepseek_api_url", "https://api.deepseek.com")

# 設定 OpenAI 用戶端 (如果金鑰存在)
if OPENAI_API_KEY:
    # 使用 v1.x API，不需要手動設定 openai.api_key
    # client = OpenAI() 會自動讀取環境變數
    pass # 不需額外設定
else:
    print("警告：未找到 OPENAI_API_KEY 環境變數。將無法使用 OpenAI 模型。")

# 設定 Google GenAI 用戶端 (如果金鑰存在)
if GOOGLE_API_KEY:
    # 使用新版 Client API
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        print(f"Google GenAI 客戶端初始化成功，使用 API 金鑰: {GOOGLE_API_KEY[:5]}***")
    except Exception as e:
        print(f"警告：初始化 Google GenAI 客戶端發生錯誤: {e}")
else:
    print("警告：未找到 GOOGLE_API_KEY 環境變數。將無法使用 Google Gemini 模型。")

# 設定 Ollama 用戶端
if OLLAMA_URL:
    # 檢查 URL 格式
    if OLLAMA_URL.startswith('http://') or OLLAMA_URL.startswith('https://'):
        # 如果 URL 包含 /api/chat，取出基礎 URL 部分
        if '/api/chat' in OLLAMA_URL:
            base_url = OLLAMA_URL.split('/api/chat')[0]
            print(f"Ollama API 端點: {OLLAMA_URL}")
            print(f"Ollama 服務基礎 URL: {base_url}")
        else:
            base_url = OLLAMA_URL
            print(f"Ollama 服務 URL: {base_url}")
        
        # 測試 Ollama API 是否可用
        try:
            test_url = f"{base_url}/api/tags"
            print(f"正在測試 Ollama API 連接性: {test_url}")
            response = requests.get(test_url)
            if response.status_code == 200:
                print("Ollama API 可用！")
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models] if models else []
                print(f"已安裝的模型: {model_names}")
            else:
                print(f"警告: Ollama API 回應代碼 {response.status_code}: {response.text}")
        except Exception as e:
            print(f"警告: 無法連接到 Ollama API: {str(e)}")
    else:
        print(f"警告: Ollama URL 格式不正確 (應以 http:// 或 https:// 開頭): {OLLAMA_URL}")
else:
    print("警告：未找到 ollama_url 環境變數。將無法使用 Ollama 模型。")

# --- 本地模型管理 (修改) ---
# 不在啟動時載入特定模型，改為動態載入
# 可以快取已載入的模型以提高效率
loaded_hf_models = {}
# 預設的 tiktoken 編碼器，用於本地模型 token 估算 (可以根據模型調整)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"警告：無法載入 tiktoken 編碼器 cl100k_base: {e}。本地模型的 token 計數將是粗略估計。")
    tokenizer = None


def get_hf_model(model_name: str):
    """動態載入或從快取取得 Hugging Face Sentence Transformer 模型"""
    if model_name not in loaded_hf_models:
        print(f"首次載入 Hugging Face 模型: {model_name}")
        try:
            # 確保 sentence-transformers 已安裝
            from sentence_transformers import SentenceTransformer
            loaded_hf_models[model_name] = SentenceTransformer(model_name)
            print(f"模型 {model_name} 載入成功。")
        except ImportError:
             print("錯誤: sentence-transformers 函式庫未安裝。請執行 pip install sentence-transformers")
             raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 sentence-transformers 函式庫")
        except Exception as e:
            print(f"錯誤：無法載入 Hugging Face 模型 '{model_name}'. {e}")
            # 檢查是否是常見的網路或模型不存在問題
            if "not found on HuggingFace Hub" in str(e) or "Connection Error" in str(e):
                 raise HTTPException(status_code=400, detail=f"無法找到或下載指定的 Hugging Face 模型: {model_name}")
            else:
                 raise HTTPException(status_code=500, detail=f"載入 Hugging Face 模型時發生內部錯誤: {model_name}")
    return loaded_hf_models[model_name]

def count_tokens(texts: List[str]) -> int:
    """使用 tiktoken 估算 token 數量"""
    if tokenizer:
        try:
            return sum(len(tokenizer.encode(text)) for text in texts)
        except Exception as e:
             print(f"使用 tiktoken 計算 token 時出錯: {e}. 回退到基本計算。")
             # 回退到簡單的字數統計
             return sum(len(text.split()) for text in texts)
    else:
        # 回退到簡單的字數統計
        return sum(len(text.split()) for text in texts)


# --- FastAPI 應用程式 (保持不變) ---
app = FastAPI(
    title="OpenAI-Compatible API Server",
    description="提供與 OpenAI API 相容的介面，支援 Embedding 和 Chat Completion，並整合多個 LLM 供應商。",
    version="0.3.0",
)

# --- API 端點 (大幅修改) ---

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, _authorized: bool = Depends(verify_api_key)):
    """
    根據請求的模型名稱，使用 Hugging Face、OpenAI 或 Google Gemini 產生 embedding。
    """
    start_time = time.time()
    requested_model = request.model
    input_texts = [request.input] if isinstance(request.input, str) else request.input

    # 基本輸入驗證
    if not input_texts or not all(isinstance(text, str) and text.strip() for text in input_texts):
         raise HTTPException(status_code=400, detail="輸入必須是非空的字串或非空字串列表。")

    response_data: List[EmbeddingData] = []
    usage = Usage()
    actual_model_name = requested_model # 預設回傳請求的模型名

    try:
        # --- OpenAI 模型 ---
        if requested_model.startswith("text-embedding-"):
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="伺服器未設定 OpenAI API 金鑰 (OPENAI_API_KEY)。")
            try:
                print(f"使用 OpenAI API，模型: {requested_model}")
                # 使用 v1.x SDK
                from openai import OpenAI, APIError
                client = OpenAI() # 自動讀取 OPENAI_API_KEY

                response = client.embeddings.create(
                    input=input_texts,
                    model=requested_model
                )
                # 提取數據
                for i, data in enumerate(response.data):
                    response_data.append(EmbeddingData(embedding=data.embedding, index=data.index))
                usage = Usage(prompt_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens)
                actual_model_name = response.model # 使用 OpenAI 回傳的精確模型名稱

            except APIError as e:
                 print(f"OpenAI API 錯誤: Status={e.status_code}, Message={e.message}")
                 raise HTTPException(status_code=e.status_code or 500, detail=f"OpenAI API 錯誤: {e.message}")
            except ImportError:
                print("錯誤: openai 函式庫未安裝。請執行 pip install openai")
                raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 openai 函式庫")
            except Exception as e:
                 print(f"呼叫 OpenAI 時發生未知錯誤: {e}")
                 raise HTTPException(status_code=500, detail=f"呼叫 OpenAI 時發生錯誤: {str(e)}")

        # --- Google Gemini 模型 ---
        elif requested_model.startswith("models/embedding-") or \
             requested_model.startswith("models/gemini-") or \
             (requested_model.startswith("embedding-") and not requested_model.startswith("text-embedding-")) or \
             requested_model.startswith("gemini-"):

            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="伺服器未設定 Google API 金鑰 (GOOGLE_API_KEY)。")

            # 確保傳遞給 Google API 的模型名稱包含 "models/" 前綴
            google_api_model_name = requested_model
            if not google_api_model_name.startswith("models/"):
                google_api_model_name = f"models/{google_api_model_name}"

            try:
                print(f"使用 Google GenAI API，請求模型: {requested_model} (API 模型: {google_api_model_name})")

                try:
                    import google.generativeai as genai
                except ImportError:
                     print("錯誤: google-generativeai 函式庫未安裝。請執行 pip install google-generativeai")
                     raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 google-generativeai 函式庫")

                result = genai.embed_content(
                    model=google_api_model_name, # 使用處理過的模型名稱
                    content=input_texts,
                    task_type="RETRIEVAL_DOCUMENT" # 或者根據模型選擇更合適的 task_type
                )

                if 'embedding' in result and isinstance(result['embedding'], list):
                     embeddings_list = result['embedding']
                     if len(embeddings_list) == len(input_texts):
                         for i, emb in enumerate(embeddings_list):
                             response_data.append(EmbeddingData(embedding=emb, index=i))
                     else:
                          print(f"警告：Google API 回傳的 embedding 數量 ({len(embeddings_list)}) 與輸入數量 ({len(input_texts)}) 不符。")
                          for i, emb in enumerate(embeddings_list):
                              if i < len(input_texts):
                                 response_data.append(EmbeddingData(embedding=emb, index=i))
                else:
                     print(f"Google API 回應格式非預期: {result}")
                     raise HTTPException(status_code=500, detail="無法從 Google API 回應中解析 embedding。")

                tokens = count_tokens(input_texts)
                usage = Usage(prompt_tokens=tokens, total_tokens=tokens)
                actual_model_name = requested_model # 回傳使用者請求的原始模型名稱

            except Exception as e:
                print(f"Google GenAI API 錯誤 ({google_api_model_name}): {e}")
                # 這裡可以根據 Google API 的具體錯誤類型做更細緻的處理
                # 檢查是否為模型不存在的錯誤 (這部分依賴 Google API client 的錯誤回報方式)
                if "not found" in str(e).lower() or "permission denied" in str(e).lower() or "404" in str(e):
                    raise HTTPException(status_code=404, detail=f"Google GenAI 模型 '{google_api_model_name}' 未找到或無法存取。錯誤: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Google GenAI API 錯誤 ({google_api_model_name}): {str(e)}")

        # --- Ollama 模型 ---
        elif requested_model.startswith("ollama/"):
            if not OLLAMA_URL:
                raise HTTPException(status_code=500, detail="伺服器未設定 Ollama 服務位址 (ollama_url)。")
            
            # 從模型名稱中移除 "ollama/" 前綴
            ollama_model_name = requested_model.replace("ollama/", "", 1)
            if not ollama_model_name:
                raise HTTPException(status_code=400, detail="Ollama 模型名稱格式不正確，應為 'ollama/modelname'。")

            try:
                # 處理 Ollama URL
                if '/api/chat' in OLLAMA_URL:
                    base_url = OLLAMA_URL.split('/api/chat')[0]
                else:
                    base_url = OLLAMA_URL.rstrip('/')
                
                print(f"使用 Ollama API，請求模型: {requested_model} (實際 Ollama 模型: {ollama_model_name}), URL: {base_url}")
                
                # 設定 Ollama 客戶端，根據是否有 API 金鑰來決定是否加入認證
                headers = {}
                if OLLAMA_API_KEY:
                    headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
                
                client = ollama.Client(host=base_url, headers=headers)
                
                for i, text_input in enumerate(input_texts):
                    ollama_response = client.embeddings(
                        model=ollama_model_name,
                        prompt=text_input
                    )
                    if 'embedding' not in ollama_response or not isinstance(ollama_response['embedding'], list):
                        print(f"Ollama API 回應格式非預期 for input '{text_input[:50]}...': {ollama_response}")
                        raise HTTPException(status_code=500, detail=f"無法從 Ollama API 回應中解析 '{ollama_model_name}' 的 embedding。")
                    
                    response_data.append(EmbeddingData(embedding=ollama_response['embedding'], index=i))

                tokens = count_tokens(input_texts)
                usage = Usage(prompt_tokens=tokens, total_tokens=tokens)
                actual_model_name = requested_model

            except OllamaResponseError as e:
                detail = f"Ollama API 錯誤 ({ollama_model_name}): {str(e)}"
                status_code = 500
                if hasattr(e, 'status_code') and e.status_code:
                    status_code = e.status_code
                    if e.status_code == 404:
                        detail = f"Ollama 模型 '{ollama_model_name}' 未在 Ollama 服務 ({base_url}) 中找到。"
                    elif e.status_code == 401:
                        detail = f"Ollama API 認證失敗。請檢查 API 金鑰設定。"
                
                if "connection refused" in str(e).lower() or \
                   "Temporary failure in name resolution" in str(e).lower() or \
                   "Errno 111" in str(e) or "Errno -3" in str(e):
                    detail = f"無法連接到 Ollama 服務於 {base_url}。請確認 Ollama 服務正在運行且 URL 設定正確。"
                    status_code = 503

                print(f"Ollama API 錯誤: Status={status_code}, Detail={detail}")
                raise HTTPException(status_code=status_code, detail=detail)
            except ImportError:
                print("錯誤: ollama 函式庫未安裝。請執行 pip install ollama")
                raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 ollama 函式庫")
            except Exception as e:
                print(f"呼叫 Ollama ({ollama_model_name}) 時發生未知錯誤: {e}")
                if "connection refused" in str(e).lower() or \
                   "Temporary failure in name resolution" in str(e).lower() or \
                   "Errno 111" in str(e) or "Errno -3" in str(e):
                     raise HTTPException(status_code=503, detail=f"無法連接到 Ollama 服務於 {base_url}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"處理 Ollama embedding '{ollama_model_name}' 時發生錯誤: {str(e)}")

        # --- Hugging Face 本地模型 ---
        elif requested_model.startswith("huggingface/"):
            # 使用明確的 'huggingface/' 前綴來標識 Hugging Face 模型
            # 例如: 'huggingface/sentence-transformers/all-MiniLM-L6-v2'
            # 或是: 'huggingface/bge-large-zh-v1.5'
            
            try:
                # 移除 'huggingface/' 前綴以獲取實際模型名稱
                hf_model_name = requested_model.replace('huggingface/', '', 1)
                print(f"使用 Hugging Face 模型: {hf_model_name}")
                
                hf_model = get_hf_model(hf_model_name) # 可能拋出 HTTPException
                embeddings = hf_model.encode(input_texts, convert_to_list=True)

                for i, emb in enumerate(embeddings):
                    response_data.append(EmbeddingData(embedding=emb, index=i))

                # 計算 token
                tokens = count_tokens(input_texts)
                usage = Usage(prompt_tokens=tokens, total_tokens=tokens)
                actual_model_name = requested_model

            except HTTPException as e: # 重新拋出 get_hf_model 或其他地方的 HTTP 錯誤
                raise e
            except Exception as e:
                print(f"處理 Hugging Face embedding 時發生錯誤 ({hf_model_name}): {e}")
                # 可以判斷是否為模型不存在等特定錯誤
                raise HTTPException(status_code=500, detail=f"處理 Hugging Face embedding '{hf_model_name}' 時發生錯誤: {str(e)}")

        # --- 其他模型 (向後兼容性) ---
        else:
            # 假設其他所有模型名稱也都是 Hugging Face 模型 (向後兼容)
            try:
                print(f"嘗試使用未指定前綴的模型: {requested_model}，假設為 Hugging Face 模型")
                hf_model = get_hf_model(requested_model) # 可能拋出 HTTPException
                embeddings = hf_model.encode(input_texts, convert_to_list=True)

                for i, emb in enumerate(embeddings):
                    response_data.append(EmbeddingData(embedding=emb, index=i))

                # 計算 token
                tokens = count_tokens(input_texts)
                usage = Usage(prompt_tokens=tokens, total_tokens=tokens)
                actual_model_name = requested_model

            except HTTPException as e: # 重新拋出 get_hf_model 或其他地方的 HTTP 錯誤
                raise e
            except Exception as e:
                print(f"處理本地 embedding 時發生錯誤 ({requested_model}): {e}")
                # 可以判斷是否為模型不存在等特定錯誤
                raise HTTPException(status_code=500, detail=f"處理本地 embedding '{requested_model}' 時發生錯誤: {str(e)}")

        # --- 建立最終回應 ---
        final_response = EmbeddingResponse(
            data=response_data,
            model=actual_model_name,
            usage=usage
        )

        end_time = time.time()
        print(f"模型 '{actual_model_name}' 請求處理完成 ({len(input_texts)} 輸入)，耗時：{end_time - start_time:.4f} 秒")

        return final_response

    except HTTPException as e: # 捕獲上面明確拋出的 HTTP 異常
        raise e # 直接重新拋出
    except Exception as e: # 捕獲未預料的全局錯誤
        print(f"處理 embedding 請求 '{requested_model}' 時發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印詳細追蹤信息以供調試
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤，請檢查伺服器日誌。")

# --- Chat Completion 處理函數 ---
async def stream_chat_completion_openai(request: ChatCompletionRequest):
    if request.model.startswith("openai/"):
        # --- Existing OpenAI Logic ---
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="未設定 OpenAI API 金鑰")
        try:
            model_name = request.model.replace('openai/', '')
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            stream = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                stream=True
            )
            
            async def generate_openai_stream():
                try:
                    async for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            response_data = {
                                'choices': [{
                                    'delta': {
                                        'content': chunk.choices[0].delta.content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_msg = f"OpenAI 串流處理錯誤: {str(e)}"
                    print(error_msg)
            
            return StreamingResponse(
                generate_openai_stream(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream; charset=utf-8"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API 錯誤: {str(e)}")

    elif request.model.startswith("deepseek/"):
        # --- New DeepSeek Logic ---
        if not DEEPSEEK_API_KEY:
            raise HTTPException(status_code=500, detail="未設定 DeepSeek API 金鑰")
        
        print("DeepSeek stream: Matched deepseek model for streaming.") # Log path matching

        model_name = request.model.replace('deepseek/', '')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY.strip()}"
        }
        payload_data = {
            "model": model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "stream": True
        }
        if request.max_tokens is not None:
            payload_data["max_tokens"] = request.max_tokens
        
        async def generate_deepseek_stream_content():
            print("DeepSeek stream: >>> ENTERING generate_deepseek_stream_content <<<") # Simplest entry log
            server_sent_done = False
            try:
                print(f"DeepSeek stream: Sending request to {DEEPSEEK_API_URL}/chat/completions with payload: {json.dumps(payload_data)}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{DEEPSEEK_API_URL}/chat/completions",
                        headers=headers,
                        json=payload_data,
                        timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
                    ) as response:
                        print(f"DeepSeek stream: Connection attempt made. Response status from DeepSeek: {response.status}") # Log status immediately
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"DeepSeek API 請求失敗 ({response.status}): {error_text}")
                            return

                        print(f"DeepSeek stream: Successfully connected to DeepSeek. Response Headers: {response.headers}") # Log headers on success
                        
                        any_line_processed = False
                        async for line_bytes in response.content:
                            any_line_processed = True # Mark that we entered the loop
                            line_bytes = line_bytes.strip()
                            if not line_bytes:
                                continue
                            
                            line_str = ""
                            try:
                                line_str = line_bytes.decode('utf-8').strip()
                                print(f"DeepSeek raw stream line: '{line_str}'")

                                if not line_str:
                                    continue

                                if line_str.startswith("data:"):
                                    json_payload_str = line_str[len("data:"):].strip()
                                    if json_payload_str == "[DONE]":
                                        print("DeepSeek stream: Server sent event 'data: [DONE]'")
                                        server_sent_done = True
                                        break
                                    if not json_payload_str:
                                        print("DeepSeek stream: Received 'data: ' followed by empty payload, skipping.")
                                        continue
                                    try:
                                        chunk_data = json.loads(json_payload_str)
                                        ds_choices = chunk_data.get("choices")
                                        if ds_choices and len(ds_choices) > 0:
                                            ds_delta = ds_choices[0].get("delta", {})
                                            ds_content = ds_delta.get("content")
                                            if ds_content:
                                                response_data = {'choices': [{'delta': {'content': ds_content}}]}
                                                print(f"DeepSeek stream: YIELDING to client: {json.dumps(response_data, ensure_ascii=False)}") # Log before yield
                                                yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                    except json.JSONDecodeError as e_json:
                                        print(f"DeepSeek stream: JSON parsing failed for 'data: ' payload: '{json_payload_str}'. Error: {e_json}")
                                        continue
                                else:
                                    print(f"DeepSeek stream: Received non-SSE-data line: '{line_str}'")
                            except UnicodeDecodeError as e_decode:
                                print(f"DeepSeek stream: UTF-8 decoding failed for bytes: {line_bytes[:100]}... Error: {e_decode}")
                                continue
                            except Exception as e_line_proc:
                                print(f"DeepSeek stream: Error processing line '{line_str}': {str(e_line_proc)}")
                                continue
                        
                        if not any_line_processed:
                            print("DeepSeek stream: No lines were processed from response.content. The stream might be empty or DeepSeek closed it prematurely.")
                            # Optional: Attempt to read a small part if stream was empty to see if there's any non-streaming error message
                            try:
                                error_body_preview = await response.text() # .text() consumes, use carefully or if sure stream is empty
                                print(f"DeepSeek stream: Fallback read of response body (if stream was empty): {error_body_preview[:1000]}") # Log first 1000 chars
                            except Exception as e_read_fallback:
                                print(f"DeepSeek stream: Error during fallback read of response body: {e_read_fallback}")

            except aiohttp.ClientError as e:
                print(f"DeepSeek stream (aiohttp.ClientError during connection or request): {str(e)}")
            except asyncio.TimeoutError:
                print(f"DeepSeek stream (TimeoutError)")
            except Exception as e:
                print(f"DeepSeek stream (general error in generate_deepseek_stream_content): {str(e)}")
            finally:
                if not server_sent_done:
                    print("DeepSeek stream: Server did not send 'data: [DONE]', sending client 'data: [DONE]' in finally block.")
                    yield "data: [DONE]\n\n"
                else:
                    print("DeepSeek stream: Server sent 'data: [DONE]', not sending another from finally block.")
        
        # This part is outside the generator, in the main function body for deepseek branch
        try:
            print("DeepSeek stream: About to return StreamingResponse for DeepSeek.") # Log before StreamingResponse
            return StreamingResponse(
                generate_deepseek_stream_content(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream; charset=utf-8"}
            )
        except Exception as e:
            print(f"DeepSeek stream: ERROR creating StreamingResponse for DeepSeek: {str(e)}") # Log error during StreamingResponse creation
            raise HTTPException(status_code=500, detail=f"DeepSeek API 串流設定錯誤: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail=f"模型 '{request.model}' 的串流處理不支援。")

async def stream_chat_completion_gemini(request: ChatCompletionRequest):
    """處理 Google Gemini 的串流聊天完成請求"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 Google API 金鑰")
    
    try:
        # 確保引入所需模組
        import google.generativeai as genai_lib
        from google.genai import types
        
        # 移除 'gemini/' 前綴並加上 'models/' 前綴（如果需要）
        model_name = request.model.replace('gemini/', '')
        if not model_name.startswith('models/'):
            model_name = f"models/{model_name}"
        
        print(f"使用 Gemini 串流模型: {model_name}")
        
        # 使用自定義生成器來確保立即發送每個串流片段
        async def generate_with_immediate_yield():
            try:
                # 處理請求訊息
                formatted_messages = []
                system_instruction = None
                
                for msg in request.messages:
                    if msg.role == "system":
                        system_instruction = msg.content
                    elif msg.role == "user":
                        formatted_messages.append({"role": "user", "parts": [{"text": msg.content}]})
                    elif msg.role == "assistant":
                        formatted_messages.append({"role": "model", "parts": [{"text": msg.content}]})
                
                print(f"系統指令: {system_instruction}")
                print(f"訊息數量: {len(formatted_messages)}")
                
                # 建立不包含 stream 參數的配置對象
                config = types.GenerateContentConfig(
                    temperature=request.temperature
                )
                
                if system_instruction:
                    config.system_instruction = system_instruction
                
                if request.max_tokens:
                    config.max_output_tokens = request.max_tokens
                
                # 使用 generate_content_stream 方法
                print("使用 client.models.generate_content_stream 進行真實串流")
                
                # 使用示例代碼中的方式，直接採用 generate_content_stream
                response_stream = await asyncio.to_thread(
                    client.models.generate_content_stream,
                    model=model_name,
                    contents=formatted_messages,
                    config=config
                )
                
                # 處理串流回應，確保每個片段立即發送
                print("開始處理串流回應並立即發送片段")
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text and chunk.text.strip():
                        print(f"獲取片段: {chunk.text}")
                        response_data = {
                            'choices': [{
                                'delta': {
                                    'content': chunk.text
                                }
                            }]
                        }
                        data = f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                        print(f"發送片段: {len(data)} 字節")
                        yield data
                        # 加入小延遲確保客戶端有時間處理
                        await asyncio.sleep(0.01)
                
                # 發送結束標記
                print("發送 [DONE] 標記")
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = f"Gemini 串流處理錯誤: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # 在錯誤時也發送 [DONE] 信號
                yield f"data: {{\"error\": \"{error_msg}\"}}\n\n"
                yield "data: [DONE]\n\n"
        
        # 返回串流響應，使用特殊設定確保即時發送
        return StreamingResponse(
            generate_with_immediate_yield(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用 Nginx 緩衝
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Gemini API 串流處理請求時發生意外錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gemini API 錯誤: {str(e)}")

async def stream_chat_completion_ollama(request: ChatCompletionRequest):
    """處理 Ollama 的串流聊天完成請求"""
    if not OLLAMA_URL:
        raise HTTPException(status_code=500, detail="未設定 Ollama 服務位址")
    
    try:
        # 移除 'ollama/' 前綴
        model_name = request.model.replace('ollama/', '')
        
        # 準備請求資料
        data = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": {
                "temperature": request.temperature
            }
        }
        
        async def generate():
            try:
                # 設定較長的超時時間
                timeout = aiohttp.ClientTimeout(total=300)  # 5分鐘
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # 確定正確的 API 端點 URL
                    api_url = OLLAMA_URL
                    
                    # 檢查 URL 是否已經包含 /api/chat 路徑
                    if not (api_url.endswith('/api/chat') or api_url.endswith('/api/chat/')):
                        # 如果不包含，則構建完整的 API 路徑
                        if api_url.endswith('/'):
                            api_url = f"{api_url}api/chat" 
                        else:
                            api_url = f"{api_url}/api/chat"
                    
                    print(f"Ollama 串流請求 URL: {api_url}")
                    
                    async with session.post(
                        api_url,
                        json=data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Ollama API 錯誤: {error_text}"
                            )
                        
                        async for line in response.content:
                            if line:
                                try:
                                    # 解析 Ollama 的回應
                                    chunk = json.loads(line)
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        # 使用 ensure_ascii=False 來正確處理中文字符
                                        response_data = {
                                            'choices': [{
                                                'delta': {
                                                    'content': chunk['message']['content']
                                                }
                                            }]
                                        }
                                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError as e:
                                    print(f"JSON 解析錯誤: {str(e)}")
                                    continue
                                except KeyError as e:
                                    print(f"回應格式錯誤: {str(e)}")
                                    continue
                yield "data: [DONE]\n\n"
            except asyncio.TimeoutError:
                error_msg = "Ollama API 請求超時"
                print(error_msg)
                raise HTTPException(status_code=504, detail=error_msg)
            except aiohttp.ClientError as e:
                error_msg = f"Ollama API 連線錯誤: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            except Exception as e:
                error_msg = f"Ollama API 錯誤: {str(e)}"
                print(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API 錯誤: {str(e)}")

async def stream_chat_completion_minmax(request: ChatCompletionRequest):
    """處理 MinMax 的串流聊天完成請求"""
    if not MINMAX_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 MinMax API 金鑰 (MINMAX_API_KEY)")
    if not MINMAX_API_URL:
        # MINMAX_API_URL is used as the full endpoint for both stream and non-stream
        # It's typically defined as os.getenv("minmax_baseurl", "https://api.minimax.chat/v1/text/chatcompletion_v2")
        raise HTTPException(status_code=500, detail="未設定 MinMax API URL (minmax_baseurl)")

    print("MinMax stream: Matched minmax model for streaming.")

    model_name = request.model.replace("minmax/", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINMAX_API_KEY.strip()}"
    }
    
    # Construct payload for MinMax API
    # Note: MinMax might have a slightly different payload structure for messages or stream options.
    # This is a generic structure based on OpenAI compatibility.
    # Refer to MinMax documentation for exact payload requirements if issues arise.
    payload_data = {
        "model": model_name,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature,
        "stream": True, # Crucial for streaming
        # MinMax might use "tokens_to_generate" instead of "max_tokens"
        # MinMax also has other parameters like "top_p", "temperature", "mask_sensitive_info" etc.
        # Add them here if needed based on MinMax API docs and what ChatCompletionRequest supports
    }
    if request.max_tokens is not None:
        # Assuming MinMax uses 'max_tokens', if not, this needs to be mapped to their equivalent
        payload_data["max_tokens"] = request.max_tokens 
        # Or if MinMax uses "tokens_to_generate":
        # payload_data["tokens_to_generate"] = request.max_tokens

    async def generate_minmax_stream_content():
        print(f"MinMax stream: >>> ENTERING generate_minmax_stream_content for model {model_name} <<<")
        server_sent_done = False
        try:
            print(f"MinMax stream: Sending request to {MINMAX_API_URL} with payload: {json.dumps(payload_data)}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    MINMAX_API_URL, 
                    headers=headers,
                    json=payload_data,
                    timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT) 
                ) as response:
                    print(f"MinMax stream: Connection attempt made. Response status from MinMax: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"MinMax API 請求失敗 ({response.status}): {error_text}")
                        return

                    print(f"MinMax stream: Successfully connected to MinMax. Response Headers: {response.headers}")
                    
                    any_line_processed = False
                    async for line_bytes in response.content:
                        any_line_processed = True
                        line_bytes = line_bytes.strip()
                        if not line_bytes:
                            continue
                        
                        line_str = ""
                        try:
                            line_str = line_bytes.decode('utf-8').strip()
                            print(f"MinMax raw stream line: '{line_str}'")

                            if not line_str:
                                continue
                            
                            # MinMax SSE format might be: data: {...json...}
                            # Or it could be just JSON lines without the "data: " prefix.
                            # The provided non-streaming example returns a JSON with "base_resp" and "reply".
                            # Streaming might be different. Assuming standard SSE "data: " prefix for now.

                            if line_str.startswith("data:"):
                                json_payload_str = line_str[len("data:"):].strip()
                                
                                if json_payload_str == "[DONE]":
                                    print("MinMax stream: Server sent event 'data: [DONE]'")
                                    server_sent_done = True
                                    break

                                if not json_payload_str:
                                    print("MinMax stream: Received 'data: ' followed by empty payload, skipping.")
                                    continue

                                try:
                                    chunk_data = json.loads(json_payload_str)
                                    # Adapt parsing to actual MinMax stream chunk structure
                                    # Example: {"request_id":"...","created":...,"model":"...","choices":[{"delta":{"content":"..."}}]}
                                    # Or it might be {"data": {"request_id": ..., "choices": [...] } ... } if nested further
                                    mm_choices = chunk_data.get("choices")
                                    mm_content = None
                                    if mm_choices and len(mm_choices) > 0:
                                        # Check if delta exists, or if content is directly in choice
                                        if "delta" in mm_choices[0] and isinstance(mm_choices[0]["delta"], dict):
                                            mm_content = mm_choices[0]["delta"].get("content")
                                        elif "text" in mm_choices[0]: # Alternative common field name
                                            mm_content = mm_choices[0]["text"]
                                        # Add other possible field names if MinMax uses something different

                                    if mm_content is not None:
                                        response_data = {'choices': [{'delta': {'content': mm_content}}]}
                                        print(f"MinMax stream: YIELDING to client: {json.dumps(response_data, ensure_ascii=False)}")
                                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError as e_json:
                                    print(f"MinMax stream: JSON parsing failed for 'data: ' payload: '{json_payload_str}'. Error: {e_json}")
                                    continue
                            else:
                                # If MinMax sends JSON lines without "data: " prefix:
                                try:
                                    chunk_data = json.loads(line_str) # Try parsing the whole line as JSON
                                    mm_choices = chunk_data.get("choices")
                                    mm_content = None
                                    if mm_choices and len(mm_choices) > 0:
                                        if "delta" in mm_choices[0] and isinstance(mm_choices[0]["delta"], dict):
                                            mm_content = mm_choices[0]["delta"].get("content")
                                        elif "text" in mm_choices[0]:
                                            mm_content = mm_choices[0]["text"]
                                    
                                    if mm_content is not None:
                                        response_data = {'choices': [{'delta': {'content': mm_content}}]}
                                        print(f"MinMax stream (direct JSON line): YIELDING to client: {json.dumps(response_data, ensure_ascii=False)}")
                                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                    # Check for a [DONE] equivalent if not using SSE standard
                                    # For example, if a specific flag indicates end of stream in the JSON.
                                    # if chunk_data.get("is_finish") == True: server_sent_done = True; break;

                                except json.JSONDecodeError:
                                    print(f"MinMax stream: Received non-SSE-data line, and also not valid JSON: '{line_str}'")

                        except UnicodeDecodeError as e_decode:
                            print(f"MinMax stream: UTF-8 decoding failed for received bytes: {line_bytes[:100]}... Error: {e_decode}")
                            continue
                        except Exception as e_line_proc:
                            print(f"MinMax stream: Error processing line '{line_str}': {str(e_line_proc)}")
                            continue
                    
                    if not any_line_processed:
                        print("MinMax stream: No lines were processed from response.content.")
                        try:
                            error_body_preview = await response.text()
                            print(f"MinMax stream: Fallback read of response body: {error_body_preview[:500]}")
                        except Exception as e_read_fallback:
                            print(f"MinMax stream: Error during fallback read of response body: {e_read_fallback}")

        except aiohttp.ClientError as e_client:
            print(f"MinMax stream (aiohttp.ClientError during connection or request): {str(e_client)}")
        except asyncio.TimeoutError:
            print("MinMax stream (TimeoutError)")
        except Exception as e_general:
            print(f"MinMax stream (general error in generate_minmax_stream_content): {str(e_general)}")
        finally:
            if not server_sent_done:
                print("MinMax stream: Server did not send 'data: [DONE]', sending client 'data: [DONE]' in finally block.")
                yield "data: [DONE]\n\n"
            else:
                print("MinMax stream: Server sent 'data: [DONE]', not sending another from finally block.")

    try:
        print("MinMax stream: About to return StreamingResponse for MinMax.")
        return StreamingResponse(
            generate_minmax_stream_content(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
    except Exception as e_stream_resp:
        print(f"MinMax stream: ERROR creating StreamingResponse for MinMax: {str(e_stream_resp)}")
        raise HTTPException(status_code=500, detail=f"MinMax API 串流設定錯誤: {str(e_stream_resp)}")

async def stream_chat_completion_claude(request: ChatCompletionRequest):
    """處理 Anthropic Claude 的串流聊天完成請求"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 Anthropic API 金鑰 (ANTHROPIC_API_KEY)")

    model_name = request.model.replace("claude/", "")
    print(f"Claude stream: Using model {model_name}")

    system_prompt = None
    user_messages = []
    for msg in request.messages:
        if msg.role == "system":
            if system_prompt is None:
                system_prompt = msg.content
            else:
                print("Claude stream: Multiple system messages found. Only the first will be used.")
        else:
            user_messages.append({"role": msg.role, "content": msg.content})
    
    if not user_messages:
        raise HTTPException(status_code=400, detail="Claude API 請求至少需要一條 user/assistant 訊息。")

    try:
        # 使用 aiohttp 直接調用 Anthropic API，避免 SDK 的版本問題
        async def generate_claude_stream_content():
            print(f"Claude stream: >>> ENTERING generate_claude_stream_content for model {model_name} <<<")
            server_sent_event_count = 0
            
            try:
                # 準備請求數據
                payload = {
                    "model": model_name,
                    "messages": user_messages,
                    "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
                    "temperature": request.temperature if request.temperature is not None else 0.7,
                    "stream": True
                }
                if system_prompt:
                    payload["system"] = system_prompt
                
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                
                url = "https://api.anthropic.com/v1/messages"
                
                print(f"Claude stream: Sending direct API request to {url}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Claude API 請求失敗 ({response.status}): {error_text}")
                            raise HTTPException(status_code=response.status, detail=f"Claude API 錯誤: {error_text}")
                        
                        # 處理事件串流
                        async for line in response.content:
                            if not line.strip():
                                continue
                                
                            line_str = line.decode('utf-8').strip()
                            if not line_str.startswith('data: '):
                                continue
                                
                            data_json = line_str[6:] # 去除 'data: ' 前綴
                            
                            try:
                                data = json.loads(data_json)
                                event_type = data.get('type')
                                
                                # 處理內容塊增量
                                if event_type == 'content_block_delta':
                                    delta = data.get('delta', {})
                                    if delta.get('type') == 'text_delta' and 'text' in delta:
                                        content_delta = delta['text']
                                        response_data = {'choices': [{'delta': {'content': content_delta}}]}
                                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                        server_sent_event_count += 1
                                
                                # 處理結束事件
                                elif event_type == 'message_stop':
                                    print("Claude stream: Received message_stop event from server.")
                                    break
                                    
                            except json.JSONDecodeError:
                                print(f"Claude stream: Error parsing JSON from line: {line_str}")
                                continue
                                
            except Exception as e:
                error_message = f"Anthropic Claude API (stream) 錯誤: {str(e)}"
                print(error_message)
                try:
                    error_payload = {"error": {"message": error_message, "type": "claude_stream_error"}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                except:
                    pass
            finally:
                if server_sent_event_count > 0:
                    print(f"Claude stream: Sent {server_sent_event_count} content events. Sending [DONE].")
                else:
                    print("Claude stream: No content events were sent. Sending [DONE].")
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_claude_stream_content(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
        
    except Exception as e:
        print(f"Claude stream: ERROR creating StreamingResponse: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Claude API 串流設定錯誤: {str(e)}")

# --- API 端點 ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, _authorized: bool = Depends(verify_api_key)):
    """處理聊天完成請求"""
    try:
        # 根據模型名稱和 stream 參數選擇適當的處理函數
        if request.model.startswith("openai/"):
            if request.stream:
                return await stream_chat_completion_openai(request)
            else:
                return await get_chat_completion_openai(request)
        elif request.model.startswith("gemini/"):
            if request.stream:
                return await stream_chat_completion_gemini(request)
            else:
                return await get_chat_completion_gemini(request)
        elif request.model.startswith("ollama/"):
            if request.stream:
                return await stream_chat_completion_ollama(request)
            else:
                return await get_chat_completion_ollama(request)
        elif request.model.startswith("minmax/"):
            if request.stream:
                print("Routing to stream_chat_completion_minmax") # DEBUG LOG
                return await stream_chat_completion_minmax(request)
            else:
                print("Routing to get_chat_completion_minmax") # DEBUG LOG
                return await get_chat_completion_minmax(request)
        elif request.model.startswith("claude/"):
            if request.stream:
                print("Routing to stream_chat_completion_claude") # DEBUG LOG
                return await stream_chat_completion_claude(request)
            else:
                print("Routing to get_chat_completion_claude") # DEBUG LOG
                return await get_chat_completion_claude(request)
        elif request.model.startswith("huggingface/"):
            if request.stream:
                print("Routing to stream_chat_completion_huggingface") # DEBUG LOG
                return await stream_chat_completion_huggingface(request)
            else:
                print("Routing to get_chat_completion_huggingface") # DEBUG LOG
                return await get_chat_completion_huggingface(request)
        elif request.model.startswith("deepseek/"):
            if request.stream:
                print("Routing to stream_chat_completion_openai for DeepSeek") # stream_chat_completion_openai 內部處理 deepseek 串流
                return await stream_chat_completion_openai(request)
            else:
                print("Routing to get_chat_completion_deepseek") # 需要實現此函數
                return await get_chat_completion_deepseek(request) # 非串流 DeepSeek 處理函數
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的模型: {request.model}"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"處理請求時發生錯誤: {str(e)}")  # 加入錯誤日誌
        raise HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")

# --- 啟動伺服器 (用於本地測試) ---
if __name__ == "__main__":
    import uvicorn
    # 允許從任何來源訪問 (用於開發)
    # 在生產環境中，您可能需要更嚴格的 CORS 設定
    # 為了穩定性，尤其是在Windows上且有複雜依賴（如TensorFlow）時，建議 reload=False
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# 非串流處理函數
async def get_chat_completion_openai(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 OpenAI 的非串流聊天完成請求"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 OpenAI API 金鑰")
    
    try:
        # 移除 'openai/' 前綴以獲取實際的模型名稱
        model_name = request.model.replace('openai/', '')
        
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model_name,  # 使用實際的模型名稱
            messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False
        )
        
        return ChatCompletionResponse(
            id=response.id,
            created=int(time.time()),
            model=request.model,  # 保持原始請求中的模型名稱（包含前綴）
            choices=[{
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                },
                "finish_reason": choice.finish_reason
            } for choice in response.choices],
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API 錯誤: {str(e)}")

async def get_chat_completion_gemini(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 Google Gemini 的非串流聊天完成請求"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 Google API 金鑰")
    
    try:
        model_name = request.model.replace('gemini/', '')
        if not model_name.startswith('models/'):
            model_name = f"models/{model_name}"
        
        print(f"使用 Gemini 模型: {model_name}")
        
        system_instruction = None
        user_and_assistant_messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                user_and_assistant_messages.append({"role": msg.role, "parts": [{"text": msg.content}]})
        
        from google.genai import types
        generation_config = types.GenerateContentConfig(
            temperature=request.temperature,
            system_instruction=system_instruction if system_instruction else None 
        )
        if request.max_tokens:
            generation_config.max_output_tokens = request.max_tokens

        print(f"嘗試使用 API (非串流) - System Instruction: {system_instruction is not None}")
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_name,
            contents=user_and_assistant_messages,
            config=generation_config
        )
            
        return ChatCompletionResponse(
            id=f"gemini-{int(time.time())}",
            created=int(time.time()),
            model=request.model,  
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0, 
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

    except types.generation_types.BlockedPromptException as e:
        print(f"Gemini API 請求因安全設定被阻擋: {e}")
        raise HTTPException(status_code=400, detail=f"請求內容可能違反安全政策而被阻擋: {str(e)}")
    except types.generation_types.StopCandidateException as e:
        print(f"Gemini API 因 StopCandidateException 停止生成: {e}")
        # 這種情況下，可能仍然有部分內容生成，可以考慮是否返回
        # 這裡我們選擇返回一個錯誤，但也可以構建一個包含已生成內容的回應
        content_so_far = ""
        if hasattr(e, '__cause__') and hasattr(e.__cause__, 'last_response') and e.__cause__.last_response:
            if hasattr(e.__cause__.last_response, 'text'):
                content_so_far = e.__cause__.last_response.text
        raise HTTPException(status_code=500, detail=f"內容生成提前終止 (可能因安全或內容政策)。已生成部分: '{content_so_far or '無'}'")
    except Exception as e:
        error_message = str(e)
        print(f"Gemini API 處理非串流請求時發生錯誤: {error_message}")
        status_code = 500
        detail = f"Gemini API 錯誤: {error_message}"

        if "API key not valid" in error_message:
            status_code = 401
            detail = "Google API 金鑰無效。請檢查您的 API 金鑰設定。"
        elif "Content with system role is not supported" in error_message:
            status_code = 400
            detail = "Gemini API 目前的呼叫方式不支援直接在訊息內容中包含 system role。"
        elif "not found" in error_message.lower() or "PermissionDenied" in error_message or "404" in error_message:
            status_code = 404
            detail = f"Gemini 模型 '{model_name}' 未找到或無法存取。錯誤: {error_message}"
        elif "INVALID_ARGUMENT" in error_message:
            status_code = 400
            detail = f"Gemini API 請求參數無效: {error_message}"
        
        raise HTTPException(status_code=status_code, detail=detail)

async def get_chat_completion_ollama(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 Ollama 的非串流聊天完成請求"""
    if not OLLAMA_URL:
        raise HTTPException(status_code=500, detail="未設定 Ollama 服務位址")
    
    model_name = request.model.replace("ollama/", "")
    
    headers = {
        "Content-Type": "application/json"
    }
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            } for msg in request.messages
        ],
        "stream": False,
        "temperature": request.temperature
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # 確定正確的 API 端點 URL
            api_url = OLLAMA_URL
            
            # 檢查 URL 是否已經包含 /api/chat 路徑
            if not (api_url.endswith('/api/chat') or api_url.endswith('/api/chat/')):
                # 如果不包含，則構建完整的 API 路徑
                if api_url.endswith('/'):
                    api_url = f"{api_url}api/chat" 
                else:
                    api_url = f"{api_url}/api/chat"
            
            print(f"Ollama 請求 URL: {api_url}")
            
            async with session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=OLLAMA_TIMEOUT
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Ollama API 回應錯誤: {error_msg}"
                    )
                
                result = await response.json()
                return ChatCompletionResponse(
                    id=f"ollama-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "message": {
                            "role": "assistant",
                            "content": result["message"]["content"]
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": 0,  # Ollama 不提供 token 計數
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API 錯誤: {str(e)}")

async def get_chat_completion_minmax(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 MinMax 的非串流聊天完成請求"""
    if not MINMAX_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 MinMax API 金鑰 (MINMAX_API_KEY)")
    if not MINMAX_API_URL: # This is the full endpoint for MinMax
        raise HTTPException(status_code=500, detail="未設定 MinMax API URL (minmax_baseurl)")

    print(f"MinMax non-stream: Matched minmax model for non-streaming. Target URL: {MINMAX_API_URL}")

    model_name = request.model.replace("minmax/", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINMAX_API_KEY.strip()}"
    }
    
    payload_data = {
        "model": model_name,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature,
        "stream": False # Explicitly False for non-streaming
    }
    if request.max_tokens is not None:
        # MinMax documentation should be checked for the correct parameter name if not "max_tokens"
        # Common alternatives: "tokens_to_generate", "max_output_tokens"
        # For now, assuming "max_tokens" or that MinMax ignores it if not applicable for the model.
        # MinMax might use "tokens_to_generate"
        payload_data["max_tokens"] = request.max_tokens 

    print(f"MinMax non-stream: Sending request to {MINMAX_API_URL} with payload: {json.dumps(payload_data)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                MINMAX_API_URL, 
                headers=headers,
                json=payload_data,
                timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT) 
            ) as response:
                print(f"MinMax non-stream: Response status from MinMax: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    print(f"MinMax API (non-stream) 回應錯誤 ({response.status}): {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"MinMax API (non-stream) 回應錯誤: {error_text}"
                    )
                
                result = await response.json()
                print(f"MinMax non-stream: Received response JSON: {result}")
                
                # Validate MinMax non-stream response structure
                # Based on typical OpenAI-like or MinMax specific structure:
                # Example: { "id": "...", "created": ..., "model": "...", 
                #            "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "..."}],
                #            "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
                #            "base_resp": {"status_code": 0, "status_msg": "success"} }
                
                base_resp = result.get("base_resp")
                if base_resp and base_resp.get("status_code") != 0:
                    error_msg = base_resp.get("status_msg", "MinMax API 回報內部錯誤")
                    print(f"MinMax API (non-stream) 邏輯錯誤: {error_msg}")
                    raise HTTPException(status_code=400, detail=f"MinMax API (non-stream) 邏輯錯誤: {error_msg}")

                if not result.get("choices") or not isinstance(result["choices"], list) or len(result["choices"]) == 0:
                    raise HTTPException(status_code=500, detail="MinMax API (non-stream) 回應格式錯誤: 缺少或無效的 'choices'")
                
                first_choice = result["choices"][0]
                if not isinstance(first_choice.get("message"), dict) or not first_choice["message"].get("content") is not None:
                    # Allow empty string for content, but message and content key must exist
                    raise HTTPException(status_code=500, detail="MinMax API (non-stream) 回應格式錯誤: 缺少或無效的 'message.content'")

                return ChatCompletionResponse(
                    id=result.get("id", f"minmax-nonstream-{int(time.time())}"),
                    created=result.get("created", int(time.time())),
                    model=result.get("model", request.model),
                    choices=[{
                        "message": {
                            "role": first_choice["message"].get("role", "assistant"),
                            "content": first_choice["message"]["content"]
                        },
                        "finish_reason": first_choice.get("finish_reason", "stop")
                    }],
                    usage={
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": result.get("usage", {}).get("completion_tokens", result.get("usage", {}).get("total_tokens", 0) - result.get("usage", {}).get("prompt_tokens", 0) ), # Calculate if only total and prompt provided
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                    }
                )

    except aiohttp.ClientError as e_client:
        print(f"MinMax API (non-stream) 連線錯誤: {str(e_client)}")
        raise HTTPException(status_code=503, detail=f"MinMax API (non-stream) 連線錯誤: {str(e_client)}")
    except json.JSONDecodeError as e_json:
        print(f"MinMax API (non-stream) JSON 解析錯誤: {str(e_json)}")
        raise HTTPException(status_code=500, detail=f"MinMax API (non-stream) JSON 解析錯誤: {str(e_json)}")
    except Exception as e_general:
        print(f"MinMax API (non-stream) 未知錯誤: {str(e_general)}")
        raise HTTPException(status_code=500, detail=f"MinMax API (non-stream) 錯誤: {str(e_general)}")

async def get_chat_completion_claude(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 Anthropic Claude 的非串流聊天完成請求"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 Anthropic API 金鑰 (ANTHROPIC_API_KEY)")

    model_name = request.model.replace("claude/", "")
    print(f"Claude non-stream: Using model {model_name}")

    # 從 messages 中分離 system prompt (如果有的話)
    system_prompt = None
    user_messages = []
    for msg in request.messages:
        if msg.role == "system":
            if system_prompt is None: # Take the first system message
                system_prompt = msg.content
            else:
                # Anthropic API v1 only supports one system message.
                # Append subsequent system messages to the first user message or handle as an error.
                # For simplicity, we'll log a warning and ignore further system messages.
                print("Claude non-stream: Multiple system messages found. Only the first will be used.")
        else:
            # Convert to Anthropic message format if needed (role: "user" or "assistant")
            user_messages.append({"role": msg.role, "content": msg.content})
    
    # Anthropic SDK requires at least one message in the messages list.
    if not user_messages:
        raise HTTPException(status_code=400, detail="Claude API 請求至少需要一條 user/assistant 訊息。")

    try:
        from anthropic import Anthropic, AsyncAnthropic # Ensure SDK is available
    except ImportError:
        print("錯誤: anthropic SDK 未安裝。請執行 pip install anthropic")
        raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 anthropic SDK")

    try:
        # 使用 AsyncAnthropic Client
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        
        print(f"Claude non-stream: Sending request to Anthropic API. System prompt: '{system_prompt if system_prompt else "None"}'. Messages: {user_messages}")

        # 构建请求参数
        api_params = {
            "model": model_name,
            "messages": user_messages,
            "max_tokens": request.max_tokens if request.max_tokens is not None else 1024, # Anthropic requires max_tokens
            "temperature": request.temperature if request.temperature is not None else 0.7,
        }
        if system_prompt:
            api_params["system"] = system_prompt
        
        response = await client.messages.create(**api_params)

        print(f"Claude non-stream: Received response from Anthropic API: {response}")

        # 从 Claude 的回应中提取内容
        # Claude's response object: response.content is a list of ContentBlock objects.
        # We are interested in the text from these blocks.
        response_content = ""
        if response.content and isinstance(response.content, list):
            for block in response.content:
                if hasattr(block, 'text'): # Check if it's a TextBlock
                    response_content += block.text
        
        # Claude 的使用情况信息在 response.usage 中
        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        total_tokens = prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            id=response.id if hasattr(response, 'id') else f"claude-{int(time.time())}",
            created=int(time.time()), # Claude API doesn't provide 'created' in response object directly
            model=response.model if hasattr(response, 'model') else model_name,
            choices=[{
                "message": {
                    "role": "assistant", # Claude's response role is 'assistant'
                    "content": response_content
                },
                "finish_reason": response.stop_reason if hasattr(response, 'stop_reason') else "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )

    except ImportError: # Should have been caught above, but as a safeguard
        raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 anthropic SDK")
    except Exception as e:
        error_message = f"Anthropic Claude API (non-stream) 錯誤: {str(e)}"
        print(error_message)
        # Check for specific Anthropic API errors if possible from the exception type or message
        # e.g., if str(e) contains "authentication_error", "permission_error", "not_found_error", "rate_limit_error"
        if "authentication failed" in str(e).lower() or "invalid api key" in str(e).lower():
            raise HTTPException(status_code=401, detail=f"Anthropic API 金鑰無效或認證失敗: {str(e)}")
        if "not found" in str(e).lower() and "model" in str(e).lower():
             raise HTTPException(status_code=404, detail=f"Anthropic 模型 '{model_name}' 未找到: {str(e)}")
        # Add more specific error handling as needed
        raise HTTPException(status_code=500, detail=error_message) 

async def get_chat_completion_huggingface(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 Hugging Face 的非串流聊天完成請求"""
    try:
        # 移除 'huggingface/' 前綴
        model_name = request.model.replace('huggingface/', '', 1)
        print(f"使用 Hugging Face 聊天模型: {model_name}")
        
        # 嘗試導入必要的庫
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
        except ImportError:
            print("錯誤: transformers 或 torch 庫未安裝。請執行 pip install transformers torch")
            raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 transformers 或 torch 庫")
        
        # 加載模型與分詞器
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            print(f"已成功加載 Hugging Face 模型: {model_name}")
        except Exception as e:
            print(f"加載 Hugging Face 模型 {model_name} 失敗: {e}")
            raise HTTPException(status_code=500, detail=f"加載 Hugging Face 模型 {model_name} 失敗: {str(e)}")
        
        # 構建提示文本
        system_content = None
        prompt = ""
        
        # 處理系統提示與訊息
        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        # 如果有系統提示，加到前面
        if system_content:
            prompt = f"System: {system_content}\n" + prompt
        
        # 加入最後的助手標記，表示助手即將回應
        prompt += "Assistant: "
        
        # 準備輸入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回應
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens if request.max_tokens else 1024,
                temperature=request.temperature if request.temperature else 0.7,
                do_sample=True if request.temperature and request.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解碼回應並處理
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手的回應部分
        assistant_response = generated_text[len(prompt):]
        
        # 提取輸入與輸出 token 數量
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(outputs[0]) - input_tokens
        total_tokens = input_tokens + output_tokens
        
        return ChatCompletionResponse(
            id=f"huggingface-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        )
    except Exception as e:
        print(f"Hugging Face Chat Completion 錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hugging Face Chat Completion 錯誤: {str(e)}")

async def stream_chat_completion_huggingface(request: ChatCompletionRequest):
    """處理 Hugging Face 的串流聊天完成請求"""
    try:
        # 移除 'huggingface/' 前綴
        model_name = request.model.replace('huggingface/', '', 1)
        print(f"使用 Hugging Face 串流聊天模型: {model_name}")
        
        # 嘗試導入必要的庫
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
            from threading import Thread
        except ImportError:
            print("錯誤: transformers 或 torch 庫未安裝。請執行 pip install transformers torch")
            raise HTTPException(status_code=500, detail="伺服器錯誤：缺少 transformers 或 torch 庫")
        
        # 加載模型與分詞器
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            print(f"已成功加載 Hugging Face 模型: {model_name}")
        except Exception as e:
            print(f"加載 Hugging Face 模型 {model_name} 失敗: {e}")
            raise HTTPException(status_code=500, detail=f"加載 Hugging Face 模型 {model_name} 失敗: {str(e)}")
        
        # 構建提示文本
        system_content = None
        prompt = ""
        
        # 處理系統提示與訊息
        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        # 如果有系統提示，加到前面
        if system_content:
            prompt = f"System: {system_content}\n" + prompt
        
        # 加入最後的助手標記，表示助手即將回應
        prompt += "Assistant: "
        
        # 準備輸入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 設置串流器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=OLLAMA_TIMEOUT)
        
        # 設置生成參數
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_tokens if request.max_tokens else 1024,
            temperature=request.temperature if request.temperature else 0.7,
            do_sample=True if request.temperature and request.temperature > 0 else False,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 在後台線程中執行生成
        thread = Thread(target=lambda: model.generate(**gen_kwargs))
        thread.start()
        
        # 返回串流回應
        async def generate():
            previous_text = ""
            
            try:
                # 讀取並發送增量文本
                for new_text in streamer:
                    if new_text:
                        incremental_text = new_text
                        previous_text += incremental_text
                        
                        # 構建回應分塊
                        response_data = {
                            'choices': [{
                                'delta': {
                                    'content': incremental_text
                                }
                            }]
                        }
                        
                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                        
                # 最後發送完成標記
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = f"Hugging Face 串流處理錯誤: {str(e)}"
                print(error_msg)
                # 可能的錯誤處理
                yield f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
    except Exception as e:
        print(f"Hugging Face 串流 Chat Completion 錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hugging Face 串流 Chat Completion 錯誤: {str(e)}")

async def get_chat_completion_deepseek(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """處理 DeepSeek 的非串流聊天完成請求"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 DeepSeek API 金鑰 (DEEPSEEK_API_KEY)")
    if not DEEPSEEK_API_ENDPOINT_NONSTREAM:
        raise HTTPException(status_code=500, detail="未設定 DeepSeek API 非串流端點 (DEEPSEEK_API_ENDPOINT_NONSTREAM)")

    model_name = request.model.replace("deepseek/", "", 1)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY.strip()}"
    }
    
    payload_data = {
        "model": model_name,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "temperature": request.temperature,
        "stream": False
    }
    if request.max_tokens is not None:
        payload_data["max_tokens"] = request.max_tokens

    print(f"DeepSeek non-stream: Sending request to {DEEPSEEK_API_ENDPOINT_NONSTREAM} for model {model_name}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DEEPSEEK_API_ENDPOINT_NONSTREAM,
                headers=headers,
                json=payload_data,
                timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT) 
            ) as response:
                response_text = await response.text() # Read text first for logging
                print(f"DeepSeek non-stream: Response status: {response.status}, Response body: {response_text[:500]}") # Log first 500 chars
                
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"DeepSeek API (non-stream) 回應錯誤: {response_text}"
                    )
                
                result = json.loads(response_text) # Parse from already read text
                                
                # Validate DeepSeek non-stream response structure (assuming OpenAI-like)
                if not result.get("choices") or not isinstance(result["choices"], list) or len(result["choices"]) == 0:
                    raise HTTPException(status_code=500, detail="DeepSeek API (non-stream) 回應格式錯誤: 缺少或無效的 'choices'")
                
                first_choice = result["choices"][0]
                if not isinstance(first_choice.get("message"), dict) or first_choice["message"].get("content") is None:
                    raise HTTPException(status_code=500, detail="DeepSeek API (non-stream) 回應格式錯誤: 缺少或無效的 'message.content'")

                # Extract usage details, providing defaults if not present
                usage_data = result.get("usage", {})
                prompt_tokens = usage_data.get("prompt_tokens", 0)
                completion_tokens = usage_data.get("completion_tokens", 0)
                total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
                if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0) : # Recalculate if total is 0 but others are not
                    total_tokens = prompt_tokens + completion_tokens

                return ChatCompletionResponse(
                    id=result.get("id", f"deepseek-nonstream-{int(time.time())}"),
                    created=result.get("created", int(time.time())),
                    model=result.get("model", request.model), # Use request.model to keep prefix, or result.get("model") for API returned model
                    choices=[{
                        "message": {
                            "role": first_choice["message"].get("role", "assistant"),
                            "content": first_choice["message"]["content"]
                        },
                        "finish_reason": first_choice.get("finish_reason", "stop")
                    }],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                )

    except aiohttp.ClientError as e_client:
        print(f"DeepSeek API (non-stream) 連線錯誤: {str(e_client)}")
        raise HTTPException(status_code=503, detail=f"DeepSeek API (non-stream) 連線錯誤: {str(e_client)}")
    except json.JSONDecodeError as e_json:
        print(f"DeepSeek API (non-stream) JSON 解析錯誤: {str(e_json)} for response text: {response_text[:500]}")
        raise HTTPException(status_code=500, detail=f"DeepSeek API (non-stream) JSON 解析錯誤: {str(e_json)}")
    except Exception as e_general:
        print(f"DeepSeek API (non-stream) 未知錯誤: {str(e_general)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DeepSeek API (non-stream) 錯誤: {str(e_general)}")