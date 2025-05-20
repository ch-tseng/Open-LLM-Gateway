from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field, model_validator
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
import datetime # Add datetime
import atexit # Add atexit

# --- API 金鑰驗證依賴動態切換 ---
def always_allow():
    return True

def verify_api_key(authorization: str = Header(None)):
    whitelist = os.getenv("api_keys_whitelist", "")
    whitelist_set = {k.strip() for k in whitelist.split(",") if k.strip()}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少或格式錯誤的 API 金鑰")
    token = authorization.replace("Bearer ", "").strip()
    if token not in whitelist_set:
        raise HTTPException(status_code=401, detail="API 金鑰無效")
    return True

# 載入 .env 檔案（只需要一次）
load_dotenv(override=True) # Moved here, to be before getenv calls for ENABLE_CHECK_APIKEY

ENABLE_CHECK_APIKEY = os.getenv("ENABLE_CHECK_APIKEY", "False").lower() == "true"
api_key_dependency = Depends(verify_api_key) if ENABLE_CHECK_APIKEY else Depends(always_allow)

# --- 從請求提取 API 金鑰的輔助函數 ---
async def get_api_key_from_request(request: Request) -> Optional[str]:
    authorization: Optional[str] = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "").strip()
    return None

# --- 日誌記錄相關 (從先前版本恢復) ---
HISTORY_APIKEY_DIR = os.getenv("HISTORY_APIKEY_DIR", os.path.join(os.path.dirname(__file__), "history_apikey"))
LOG_FILE_HANDLERS = {} # To store open file handlers

def get_log_file_path(api_key: str) -> Optional[str]:
    if not api_key:
        return None
    try:
        key_log_dir = os.path.join(HISTORY_APIKEY_DIR, api_key)
        os.makedirs(key_log_dir, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        return os.path.join(key_log_dir, f"{date_str}.txt")
    except Exception as e:
        print(f"Error creating log directory for API key {api_key}: {e}")
        return None

def get_file_handler(file_path: str):
    if file_path not in LOG_FILE_HANDLERS:
        try:
            LOG_FILE_HANDLERS[file_path] = open(file_path, "a", encoding="utf-8")
        except Exception as e:
            print(f"Error opening log file {file_path}: {e}")
            return None
    return LOG_FILE_HANDLERS[file_path]

def close_all_log_files():
    print("Closing all open log files...")
    for handler in LOG_FILE_HANDLERS.values():
        try:
            handler.close()
        except Exception as e:
            print(f"Error closing a log file: {e}")
    LOG_FILE_HANDLERS.clear()

atexit.register(close_all_log_files) # Register cleanup function

def log_api_usage(
    api_key: str,
    request_type: str, # "embedding" or "chat"
    model_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    input_summary: Optional[str] = None,
    output_summary: Optional[str] = None,
    status_code: int = 200,
    error_message: Optional[str] = None
):
    if not api_key or not ENABLE_CHECK_APIKEY: # Only log if API key check is enabled and key is present
        # If api_key is a placeholder like "logging_disabled_or_no_key_needed", also skip logging.
        if api_key == "logging_disabled_or_no_key_needed":
             return
        if not ENABLE_CHECK_APIKEY: # If check is disabled, don't log based on key
             print(f"DEBUG: API Key check disabled, skipping logging for request type {request_type}, model {model_name}")
             return
        # If check is enabled but no key provided for a request that should have one,
        # it might be caught by verify_api_key. If it reaches here with no api_key,
        # it means it passed verification (e.g. verify_api_key was not applied or always_allow was used).
        # In such a case, if ENABLE_CHECK_APIKEY is true, we might still not want to log without a key.
        # However, the original logic was `if not api_key or not ENABLE_CHECK_APIKEY: return`.
        # Let's stick to that primarily.
        if not api_key: # Ensure this path is also skipped if no API key even if check is enabled.
            print(f"DEBUG: No API key provided, skipping logging for request type {request_type}, model {model_name}")
            return

    log_file_path = get_log_file_path(api_key)
    if not log_file_path:
        return

    handler = get_file_handler(log_file_path)
    if not handler:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry_parts = [
        f"[{timestamp}]",
        f"Type: {request_type}",
        f"Model: {model_name}",
        f"APIKeySuffix: ...{api_key[-4:] if api_key and len(api_key) >= 4 else 'N/A'}",
        f"PromptTokens: {prompt_tokens}",
        f"CompletionTokens: {completion_tokens}",
        f"TotalTokens: {total_tokens}",
        f"StatusCode: {status_code}",
    ]
    if input_summary:
        log_entry_parts.append(f"Input: {input_summary[:100].replace(chr(10), ' ')}")
    if output_summary:
        log_entry_parts.append(f"Output: {output_summary[:100].replace(chr(10), ' ')}")
    if error_message:
        log_entry_parts.append(f"Error: {error_message[:100].replace(chr(10), ' ')}")

    log_entry = ", ".join(log_entry_parts) + "\n"

    try:
        abs_log_path = os.path.abspath(log_file_path)
        print(f"DEBUG: Attempting to write to log file (Absolute Path): {abs_log_path}")
        print(f"DEBUG: Log entry content (first 100 chars): {log_entry[:100]}")
        handler.write(log_entry)
        handler.flush()
        print(f"DEBUG: Successfully wrote to log file (Absolute Path): {abs_log_path}")
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")

# --- Pydantic 模型定義 ---

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = None
    total_tokens: int = 0

    @model_validator(mode='after')
    def calculate_missing_tokens(cls, data: Any) -> Any:
        # data is the model instance itself in Pydantic v2 for 'after' validators
        # No need to check isinstance(data, BaseModel) usually, but it's safer with Any
        
        # Ensure prompt_tokens and total_tokens are treated as 0 if None for calculation purposes,
        # though their schema types are int, they might be set to None if coming from flexible dicts.
        # However, here they are already int=0 by default.
        prompt = data.prompt_tokens
        completion = data.completion_tokens # This one is Optional[int]
        total = data.total_tokens

        # Scenario 1: Calculate completion_tokens if missing
        if completion is None and total is not None and prompt is not None:
            # We assume total_tokens and prompt_tokens are valid for subtraction
            data.completion_tokens = total - prompt
        
        # Scenario 2: Calculate total_tokens if it's the default (0) or inconsistent
        # and others are available
        elif prompt is not None and completion is not None:
            calculated_total = prompt + completion
            # Update total_tokens if it's the default 0 or if it's inconsistent with sum
            if data.total_tokens == 0 or data.total_tokens != calculated_total:
                data.total_tokens = calculated_total
        
        # Scenario 3: (Less common) Calculate prompt_tokens if missing, and others are available
        elif prompt is None and total is not None and completion is not None:
             data.prompt_tokens = total - completion
             # And re-ensure completion_tokens isn't None if it was just calculated based on a new prompt
             if data.prompt_tokens == 0 and data.total_tokens == data.completion_tokens and data.completion_tokens is None : # Edge case if all were zero initially
                 pass # Avoid issues if all are zero

        return data

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int
    usage: Usage

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

class ChatCompletionRequestMessage(BaseModel): # <--- 更名以區分請求和回應
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionRequestMessage] # <--- 使用更名後的模型
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponseMessage(BaseModel):
    role: str
    content: Union[str, None] = None # 允許 content 為 None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# --- 環境變數與設定 (增加) ---
# load_dotenv() # 已移到 API 金鑰驗證邏輯之前

# --- API 金鑰驗證相關 (新增) ---
# ENABLE_CHECK_APIKEY = os.getenv("ENABLE_CHECK_APIKEY", "False").lower() == "true" # 已移到前面
# API_KEYS_WHITELIST_STR = os.getenv("api_keys_whitelist", "") # 相關邏輯已整合到 verify_api_key
# API_KEYS_WHITELIST = {key.strip() for key in API_KEYS_WHITELIST_STR.split(',') if key.strip()} # 相關邏輯已整合到 verify_api_key

# 讀取API金鑰白名單的路徑
API_KEYS_PATH = os.path.join(os.path.dirname(__file__), "history_apikey", "api_keys.json")

# 讀取所有環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 修改為正確的環境變數名稱
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # 修改為正確的環境變數名稱
OLLAMA_URL = os.getenv("OLLAMA_URL")  # 修改為正確的環境變數名稱
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # 修改為正確的環境變數名稱
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 修改為正確的環境變數名稱
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
DEEPSEEK_API_ENDPOINT_NONSTREAM = os.getenv("DEEPSEEK_API_ENDPOINT_NONSTREAM", "https://api.deepseek.com/chat/completions")
MINMAX_API_KEY = os.getenv("MINMAX_API_KEY")  # 修改為正確的環境變數名稱
MINMAX_API_URL = os.getenv("MINMAX_API_URL", "https://api.minimax.chat/v1/text/chatcompletion_v2")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 設定 OpenAI 和 Google 的環境變數（如果需要）
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print(f"已設定 OpenAI API 金鑰: {OPENAI_API_KEY[:5]}***")
else:
    print("警告：未找到 OPENAI_API_KEY 環境變數。將無法使用 OpenAI 模型。")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print(f"已設定 Google API 金鑰: {GOOGLE_API_KEY[:5]}***")
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
    else:
        print(f"警告: Ollama URL 格式不正確 (應以 http:// 或 https:// 開頭): {OLLAMA_URL}")
else:
    print("警告：未找到 OLLAMA_URL 環境變數。將無法使用 Ollama 模型。")

# 檢查其他 API 金鑰
if DEEPSEEK_API_KEY:
    print(f"已設定 DeepSeek API 金鑰: {DEEPSEEK_API_KEY[:5]}***")
else:
    print("警告：未找到 DEEPSEEK_API_KEY 環境變數。將無法使用 DeepSeek 模型。")

if MINMAX_API_KEY:
    print(f"已設定 MinMax API 金鑰: {MINMAX_API_KEY[:5]}***")
else:
    print("警告：未找到 MINMAX_API_KEY 環境變數。將無法使用 MinMax 模型。")

if ANTHROPIC_API_KEY:
    print(f"已設定 Anthropic API 金鑰: {ANTHROPIC_API_KEY[:5]}***")
else:
    print("警告：未找到 ANTHROPIC_API_KEY 環境變數。將無法使用 Claude 模型。")

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
async def create_embedding(request_data: EmbeddingRequest, fastapi_request: Request, _authorized: bool = api_key_dependency):
    """
    根據請求的模型名稱，使用 Hugging Face、OpenAI 或 Google Gemini 產生 embedding。
    """
    start_time = time.time()
    requested_model = request_data.model
    input_texts = [request_data.input] if isinstance(request_data.input, str) else request_data.input
    api_key_for_logging = await get_api_key_from_request(fastapi_request)
    input_summary_for_logging = "; ".join(input_texts)[:100]

    # 基本輸入驗證
    if not input_texts or not all(isinstance(text, str) and text.strip() for text in input_texts):
         raise HTTPException(status_code=400, detail="輸入必須是非空的字串或非空字串列表。")

    response_data: List[EmbeddingData] = []
    usage = Usage()
    actual_model_name = requested_model # 預設回傳請求的模型名

    try:
        # --- OpenAI 模型 ---
        if requested_model.startswith("text-embedding-") or requested_model.startswith("openai/text-embedding-"):
            if not OPENAI_API_KEY:
                exc = HTTPException(status_code=500, detail="伺服器未設定 OpenAI API 金鑰 (OPENAI_API_KEY)。")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
                raise exc
            try:
                openai_actual_model_name = requested_model
                if openai_actual_model_name.startswith("openai/"):
                    openai_actual_model_name = openai_actual_model_name.replace("openai/", "")

                print(f"使用 OpenAI API，請求模型: {requested_model} (實際 OpenAI 模型: {openai_actual_model_name})")
                # 使用 v1.x SDK
                from openai import OpenAI, APIError
                client = OpenAI() # 自動讀取 OPENAI_API_KEY

                response = client.embeddings.create(
                    input=input_texts,
                    model=openai_actual_model_name
                )
                # 提取數據
                for i, data in enumerate(response.data):
                    response_data.append(EmbeddingData(embedding=data.embedding, index=data.index))
                
                # OpenAI API v1.x SDK 的 response.usage 包含 prompt_tokens 和 total_tokens
                # usage 的型別是 openai.types.CompletionUsage
                prompt_tokens_val = response.usage.prompt_tokens if response.usage else 0
                total_tokens_val = response.usage.total_tokens if response.usage else 0
                
                usage = Usage(prompt_tokens=prompt_tokens_val, total_tokens=total_tokens_val)
                actual_model_name = requested_model # 回傳使用者請求的原始模型名稱 (包含 openai/ 前綴，如果有的話)
                
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", actual_model_name, prompt_tokens=usage.prompt_tokens, total_tokens=usage.total_tokens, input_summary=input_summary_for_logging)

            except APIError as e:
                 print(f"OpenAI API 錯誤: Status={e.status_code}, Message={e.message}")
                 exc = HTTPException(status_code=e.status_code or 500, detail=f"OpenAI API 錯誤: {e.message}")
                 if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
                 raise exc
            except ImportError:
                print("錯誤: openai 函式庫未安裝。請執行 pip install openai")
                exc = HTTPException(status_code=500, detail="伺服器錯誤：缺少 openai 函式庫")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
                raise exc
            except Exception as e:
                 print(f"呼叫 OpenAI 時發生未知錯誤: {e}")
                 exc = HTTPException(status_code=500, detail=f"呼叫 OpenAI 時發生錯誤: {str(e)}")
                 if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                 raise exc

        # --- Google Gemini 模型 ---
        elif requested_model.startswith("models/embedding-") or \
             requested_model.startswith("models/gemini-") or \
             requested_model.startswith("gemini/") or \
             (requested_model.startswith("embedding-") and not requested_model.startswith("text-embedding-")):

            if not GOOGLE_API_KEY:
                exc = HTTPException(status_code=500, detail="伺服器未設定 Google API 金鑰 (GOOGLE_API_KEY)。")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
                raise exc

            google_api_model_name = requested_model
            if google_api_model_name.startswith("gemini/"):
                google_api_model_name = google_api_model_name.replace("gemini/", "")
            
            if not google_api_model_name.startswith("models/"):
                 # Avoid adding models/ if it's already like embedding-001 (which doesn't take models/ prefix for genai typically)
                if not google_api_model_name.startswith("embedding-"): # Specific check for common non-prefixed embedding models
                    google_api_model_name = f"models/{google_api_model_name}"
                # else: for models like "embedding-001", we use them as is. The SDK handles it.

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
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", actual_model_name, prompt_tokens=tokens, total_tokens=tokens, input_summary=input_summary_for_logging)

            except Exception as e:
                print(f"Google GenAI API 錯誤 ({google_api_model_name}): {e}")
                # 這裡可以根據 Google API 的具體錯誤類型做更細緻的處理
                # 檢查是否為模型不存在的錯誤 (這部分依賴 Google API client 的錯誤回報方式)
                status_code_for_error = 500
                error_detail = f"Google GenAI API 錯誤 ({google_api_model_name}): {str(e)}"
                if "not found" in str(e).lower() or "permission denied" in str(e).lower() or "404" in str(e):
                    status_code_for_error = 404
                    error_detail = f"Google GenAI 模型 '{google_api_model_name}' 未找到或無法存取。錯誤: {str(e)}"
                exc = HTTPException(status_code=status_code_for_error, detail=error_detail)
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", google_api_model_name, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                raise exc

        # --- Ollama 模型 ---
        elif requested_model.startswith("ollama/"):
            if not OLLAMA_URL:
                exc = HTTPException(status_code=500, detail="伺服器未設定 Ollama 服務位址 (ollama_url)。")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
                raise exc
            
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
                    
                    # 為每個 embedding data 計算獨立的 usage
                    current_input_tokens = count_tokens([text_input])
                    individual_usage = Usage(prompt_tokens=current_input_tokens, total_tokens=current_input_tokens, completion_tokens=0)
                    response_data.append(EmbeddingData(embedding=ollama_response['embedding'], index=i, usage=individual_usage))

                # 整體 usage 計算 (用於 EmbeddingResponse)
                total_prompt_tokens = count_tokens(input_texts)
                usage = Usage(prompt_tokens=total_prompt_tokens, total_tokens=total_prompt_tokens, completion_tokens=0) # 確保 completion_tokens 也被設定
                actual_model_name = requested_model
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", actual_model_name, prompt_tokens=usage.prompt_tokens, total_tokens=usage.total_tokens, input_summary=input_summary_for_logging)

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
                exc = HTTPException(status_code=status_code, detail=detail)
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", ollama_model_name, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                raise exc
            except ImportError:
                print("錯誤: ollama 函式庫未安裝。請執行 pip install ollama")
                exc = HTTPException(status_code=500, detail="伺服器錯誤：缺少 ollama 函式庫")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", ollama_model_name, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message="伺服器錯誤：缺少 ollama 函式庫")
                raise exc
            except Exception as e:
                print(f"呼叫 Ollama ({ollama_model_name}) 時發生未知錯誤: {e}")
                status_code_for_error = 500
                error_detail_for_ollama = f"處理 Ollama embedding '{ollama_model_name}' 時發生錯誤: {str(e)}"
                if "connection refused" in str(e).lower() or \
                   "Temporary failure in name resolution" in str(e).lower() or \
                   "Errno 111" in str(e) or "Errno -3" in str(e):
                     status_code_for_error = 503
                     error_detail_for_ollama = f"無法連接到 Ollama 服務於 {base_url}: {str(e)}"
                exc = HTTPException(status_code=status_code_for_error, detail=error_detail_for_ollama)
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", ollama_model_name, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                raise exc

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
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", actual_model_name, prompt_tokens=tokens, total_tokens=tokens, input_summary=input_summary_for_logging)

            except HTTPException as e: # 重新拋出 get_hf_model 或其他地方的 HTTP 錯誤
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=e.status_code, error_message=e.detail)
                raise e
            except Exception as e:
                print(f"處理 Hugging Face embedding 時發生錯誤 ({hf_model_name}): {e}")
                exc = HTTPException(status_code=500, detail=f"處理 Hugging Face embedding '{hf_model_name}' 時發生錯誤: {str(e)}")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", hf_model_name, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                raise exc

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
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", actual_model_name, prompt_tokens=tokens, total_tokens=tokens, input_summary=input_summary_for_logging)

            except HTTPException as e: # 重新拋出 get_hf_model 或其他地方的 HTTP 錯誤
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=e.status_code, error_message=e.detail)
                raise e
            except Exception as e:
                print(f"處理本地 embedding 時發生錯誤 ({requested_model}): {e}")
                exc = HTTPException(status_code=500, detail=f"處理本地 embedding '{requested_model}' 時發生錯誤: {str(e)}")
                if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
                raise exc

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
        # Logging for already raised HTTPExceptions from deeper levels might be redundant if already logged there,
        # but can be added if necessary for a top-level log.
        # For now, assume deeper logs are sufficient.
        raise e # 直接重新拋出
    except Exception as e: # 捕獲未預料的全局錯誤
        print(f"處理 embedding 請求 '{requested_model}' 時發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印詳細追蹤信息以供調試
        exc = HTTPException(status_code=500, detail=f"內部伺服器錯誤，請檢查伺服器日誌。")
        if api_key_for_logging: log_api_usage(api_key_for_logging, "embedding", requested_model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
        raise exc

# --- Chat Completion 處理函數 ---
async def stream_chat_completion_openai(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model # Capture model name for logging in finally
    full_assistant_response_for_log = ""
    prompt_tokens_for_log = len(input_summary_for_logging) if input_summary_for_logging else 0
    completion_tokens_for_log = 0
    error_for_log = None
    status_code_for_log = 200

    try:
        if request.model.startswith("openai/"):
            if not OPENAI_API_KEY:
                status_code_for_log = 500
                error_for_log = "未設定 OpenAI API 金鑰"
                raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
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
                    nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
                    try:
                        async for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                content_chunk = chunk.choices[0].delta.content
                                full_assistant_response_for_log += content_chunk
                                response_data = {
                                    'choices': [{
                                        'delta': {
                                            'content': content_chunk
                                        }
                                    }]
                                }
                                yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        error_for_log = f"OpenAI 串流處理錯誤: {str(e)}"
                        status_code_for_log = 500 # Assuming internal server error for stream processing issues
                        print(error_for_log)
                        # Yield an error message if needed, or just let finally block handle logging
                        yield f"data: {json.dumps({'error': error_for_log, 'status_code': status_code_for_log}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        completion_tokens_for_log = len(full_assistant_response_for_log)
                
                return StreamingResponse(
                    generate_openai_stream(),
                    media_type="text/event-stream",
                    headers={"Content-Type": "text/event-stream; charset=utf-8"}
                )
            except Exception as e:
                status_code_for_log = 500
                error_for_log = f"OpenAI API 錯誤: {str(e)}"
                raise HTTPException(status_code=status_code_for_log, detail=error_for_log)

        elif request.model.startswith("deepseek/"):
            if not DEEPSEEK_API_KEY:
                status_code_for_log = 500
                error_for_log = "未設定 DeepSeek API 金鑰"
                raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
            
            print("DeepSeek stream: Matched deepseek model for streaming.") 

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
                nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
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
                            print(f"DeepSeek stream: Connection attempt made. Response status from DeepSeek: {response.status}") 
                            status_code_for_log = response.status # Capture status from DeepSeek
                            if response.status != 200:
                                error_text = await response.text()
                                error_for_log = f"DeepSeek API 請求失敗 ({response.status}): {error_text}"
                                print(error_for_log)
                                # No yield here, finally will log
                                return # Exit generator early

                            print(f"DeepSeek stream: Successfully connected to DeepSeek. Response Headers: {response.headers}") 
                            
                            any_line_processed = False
                            async for line_bytes in response.content:
                                any_line_processed = True 
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
                                                    full_assistant_response_for_log += ds_content
                                                    response_data = {'choices': [{'delta': {'content': ds_content}}]}
                                                    print(f"DeepSeek stream: YIELDING to client: {json.dumps(response_data, ensure_ascii=False)}") 
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
                                error_for_log = "DeepSeek stream: No lines were processed from response.content."
                                print(error_for_log)
                                # status_code_for_log might already be non-200, or it could be 200 but empty stream
                                if status_code_for_log == 200: status_code_for_log = 500 # Assume error if 200 but no data

                except aiohttp.ClientError as e:
                    error_for_log = f"DeepSeek stream (aiohttp.ClientError): {str(e)}"
                    status_code_for_log = 503 # Service unavailable
                    print(error_for_log)
                except asyncio.TimeoutError:
                    error_for_log = f"DeepSeek stream (TimeoutError)"
                    status_code_for_log = 504 # Gateway timeout
                    print(error_for_log)
                except Exception as e:
                    error_for_log = f"DeepSeek stream (general error): {str(e)}"
                    status_code_for_log = 500
                    print(error_for_log)
                finally:
                    completion_tokens_for_log = len(full_assistant_response_for_log)
                    if not server_sent_done:
                        print("DeepSeek stream: Server did not send 'data: [DONE]', sending client 'data: [DONE]' in finally block.")
                        yield "data: [DONE]\n\n"
                    else:
                        print("DeepSeek stream: Server sent 'data: [DONE]', not sending another from finally block.")
            
            try:
                print("DeepSeek stream: About to return StreamingResponse for DeepSeek.") 
                return StreamingResponse(
                    generate_deepseek_stream_content(),
                    media_type="text/event-stream",
                    headers={"Content-Type": "text/event-stream; charset=utf-8"}
                )
            except Exception as e:
                status_code_for_log = 500
                error_for_log = f"DeepSeek stream: ERROR creating StreamingResponse: {str(e)}"
                print(error_for_log) 
                raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
                
        else:
            status_code_for_log = 400
            error_for_log = f"模型 '{request.model}' 的串流處理不支援。"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
    finally:
        if api_key_for_logging: # This now executes after the StreamingResponse is potentially returned
            log_api_usage(
                api_key=api_key_for_logging,
                request_type="chat_stream_completed",
                model_name=model_name_for_log,
                prompt_tokens=prompt_tokens_for_log,
                completion_tokens=completion_tokens_for_log, # This will be 0 if stream_generator not iterated
                total_tokens=prompt_tokens_for_log + completion_tokens_for_log,
                input_summary=input_summary_for_logging,
                output_summary=full_assistant_response_for_log if full_assistant_response_for_log else None,
                status_code=status_code_for_log,
                error_message=error_for_log
            )

async def stream_chat_completion_gemini(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model 
    full_assistant_response_for_log = ""
    prompt_char_count_for_log = sum(len(m.content) for m in request.messages) if request.messages else 0
    completion_char_count_for_log = 0
    error_for_log = None
    status_code_for_log = 200

    try:
        if not GOOGLE_API_KEY:
            error_for_log = "未設定 Google API 金鑰"
            status_code_for_log = 500
        elif client is None: # client 是全域 genai.Client() 實例
            error_for_log = "Google GenAI 客戶端未初始化。"
            status_code_for_log = 500
        
        if error_for_log:
            # 如果在初始檢查中發現問題，則不繼續執行 API 呼叫邏輯
            # 錯誤將在 finally 區塊中被記錄和拋出
            pass # 允許執行流程到 finally
        else:
            # API 金鑰和 client 都有效，繼續執行
            model_name_for_api = request.model.replace('gemini/', '')
            if not model_name_for_api.startswith('models/'):
                model_name_for_api = f"models/{model_name_for_api}"
            
            print(f"使用 Gemini 串流模型: {model_name_for_api}")

            async def generate_with_immediate_yield():
                nonlocal full_assistant_response_for_log, completion_char_count_for_log, error_for_log, status_code_for_log
                try:
                    system_instruction_text = None
                    processed_messages_for_api = []
                    for msg in request.messages:
                        if msg.role == "system":
                            if system_instruction_text is None: system_instruction_text = msg.content
                            else: processed_messages_for_api.append({"role": "user", "parts": [{"text": f"(額外系統指示): {msg.content}"}]})
                        else:
                            api_role = "user" if msg.role == "user" else "model"
                            processed_messages_for_api.append({"role": api_role, "parts": [{"text": msg.content}]})
                    
                    final_contents_for_stream_api = []
                    if system_instruction_text:
                        final_contents_for_stream_api.append({"role": "user", "parts": [{"text": f"System Instruction: {system_instruction_text}\\n\\nUser Question/Input Below:"}]})
                    final_contents_for_stream_api.extend(processed_messages_for_api)

                    if not final_contents_for_stream_api or not any(part["role"] == "user" for part in final_contents_for_stream_api):
                        final_contents_for_stream_api.append({"role": "user", "parts": [{"text": "(無用戶輸入，繼續執行)"}]})
                        print("警告 (Gemini 串流): 處理後 messages 列表為空或無 user message，已添加預設 user message。")

                    stream_api_params = {
                        "model": model_name_for_api,
                        "contents": final_contents_for_stream_api,
                    }
                    # 移除 temperature 和 max_output_tokens，因為舊版 SDK 不支援直接作為參數傳遞

                    print(f"Gemini 串流 API 參數: {stream_api_params}")
                    
                    response_stream = await asyncio.to_thread(
                        client.models.generate_content_stream,
                        model=stream_api_params["model"], 
                        contents=stream_api_params["contents"]
                    )
                    
                    has_yielded_content = False
                    for chunk in response_stream:
                        # print(f"DEBUG Gemini stream chunk: {vars(chunk) if hasattr(chunk, '__dict__') else chunk}") # 再次註解掉，因為已確認 chunk 結構
                        chunk_content_to_yield = None
                        
                        # 最常見的情況：直接從 chunk.text 獲取
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_content_to_yield = chunk.text
                        # 其次，檢查 chunk.parts (這是 candidates[0].content.parts 的簡化訪問方式)
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            temp_content_from_parts = ""
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    temp_content_from_parts += part.text
                            if temp_content_from_parts:
                                chunk_content_to_yield = temp_content_from_parts
                        # 最詳細的檢查：遍歷 candidates -> content -> parts
                        elif hasattr(chunk, 'candidates') and chunk.candidates:
                            temp_content_from_candidates = ""
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'content') and candidate.content and \
                                   hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for part_item in candidate.content.parts:
                                        if hasattr(part_item, 'text') and part_item.text:
                                            temp_content_from_candidates += part_item.text
                            if temp_content_from_candidates:
                                chunk_content_to_yield = temp_content_from_candidates
                        
                        if chunk_content_to_yield:
                            content_chunk = chunk_content_to_yield
                            full_assistant_response_for_log += content_chunk
                            response_data = {'choices': [{'delta': {'content': content_chunk}}]}
                            data_to_yield = f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n" # <<< 更正這裡
                            yield data_to_yield
                            has_yielded_content = True
                            await asyncio.sleep(0.001)
                    
                    if not has_yielded_content and not error_for_log:
                        print("Gemini stream: 未產生任何內容 chunk，但串流已無錯誤地完成。")

                    yield "data: [DONE]\n\n" # <<< 更正這裡

                except Exception as e_stream_inner:
                    error_for_log = f"Gemini 串流處理內部錯誤 ({model_name_for_api}): {str(e_stream_inner)}"
                    status_code_for_log = 500 
                    print(f"Gemini 串流內部錯誤: {error_for_log}")
                    import traceback; traceback.print_exc()
                    try:
                        yield f"data: {json.dumps({'error': {'message': error_for_log, 'type': 'gemini_stream_error', 'code': status_code_for_log}}, ensure_ascii=False)}\n\n" # <<< 更正這裡
                        yield "data: [DONE]\n\n" # <<< 更正這裡
                    except Exception as e_yield_error:
                        print(f"Gemini 串流: 回報錯誤時發生錯誤: {e_yield_error}")
                finally:
                    completion_char_count_for_log = len(full_assistant_response_for_log)
            
            # 只有在初始檢查中沒有設定 error_for_log 時，才回傳 StreamingResponse
            # (generate_with_immediate_yield 內部發生的錯誤由其自身處理並記錄到 error_for_log)
            if not error_for_log:
                return StreamingResponse(
                    generate_with_immediate_yield(),
                    media_type="text/event-stream",
                    headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
                )
            # 如果 error_for_log 在初始檢查時被設定 (例如 API key 問題)，控制流將轉到外層的 finally
            # 在這種情況下，此函數不會回傳 StreamingResponse，而是依賴 finally 拋出 HTTPException
    
    # except Exception as e_outer_most_setup: # 這個 try-except 結構實際上已經包含在上面的 try 內部了
    #     # 捕捉在 'else' 區塊之外的、更早期的設定錯誤
    #     if not error_for_log: # 只有在還沒有特定錯誤時才更新
    #         error_for_log = f"Gemini 串流最外層設定錯誤: {str(e_outer_most_setup)}"
    #         status_code_for_log = 500
    #         print(f"Gemini 串流最外層設定錯誤: {error_for_log}")
    #         import traceback; traceback.print_exc()
    # 這段 except 是多餘的，因為最外層的 try 已經捕捉了所有例外

    finally:
        # 無論成功或失敗，都記錄 API 使用情況
        log_api_usage(
            api_key=api_key_for_logging,
            request_type="chat_stream_completed" if not error_for_log and status_code_for_log == 200 else "chat_stream_error",
            model_name=model_name_for_log,
            prompt_tokens=prompt_char_count_for_log, 
            completion_tokens=completion_char_count_for_log, 
            total_tokens=prompt_char_count_for_log + completion_char_count_for_log, 
            input_summary=input_summary_for_logging,
            output_summary=full_assistant_response_for_log if full_assistant_response_for_log else None,
            status_code=status_code_for_log,
            error_message=error_for_log
        )
        
        # 如果在整個 try 區塊（包括初始檢查和 API 呼叫邏輯）中發生了錯誤，
        # 並且這個錯誤應該導致 HTTP 錯誤回應，則在這裡拋出。
        # 這確保了即使 StreamingResponse 沒有被 return (因為初始錯誤)，客戶端也會收到 HTTP 錯誤。
        if error_for_log and status_code_for_log >= 400:
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)

async def stream_chat_completion_ollama(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model
    full_assistant_response_for_log = ""
    prompt_tokens_for_log = len(input_summary_for_logging) if input_summary_for_logging else 0
    completion_tokens_for_log = 0
    error_for_log = None
    status_code_for_log = 200
    try:
        if not OLLAMA_URL:
            status_code_for_log = 500
            error_for_log = "未設定 Ollama 服務位址"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
        
        try:
            model_name = request.model.replace('ollama/', '')
            data = {
                "model": model_name,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "stream": True,
                "options": {"temperature": request.temperature}
            }
            
            async def generate():
                nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
                try:
                    timeout = aiohttp.ClientTimeout(total=300)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        api_url = OLLAMA_URL
                        if not (api_url.endswith('/api/chat') or api_url.endswith('/api/chat/')):
                            api_url = f"{api_url.rstrip('/')}/api/chat"
                        
                        async with session.post(api_url, json=data, headers={"Content-Type": "application/json"}) as response:
                            status_code_for_log = response.status
                            if response.status != 200:
                                error_text = await response.text()
                                error_for_log = f"Ollama API 錯誤 ({response.status}): {error_text}"
                                # No yield, finally logs
                                return
                            
                            async for line in response.content:
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if 'message' in chunk and 'content' in chunk['message']:
                                            content_chunk = chunk['message']['content']
                                            full_assistant_response_for_log += content_chunk
                                            response_data = {'choices': [{'delta': {'content': content_chunk}}]}
                                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                    except json.JSONDecodeError as e: print(f"JSON 解析錯誤: {str(e)}"); continue
                                    except KeyError as e: print(f"回應格式錯誤: {str(e)}"); continue
                    yield "data: [DONE]\n\n"
                except asyncio.TimeoutError:
                    error_for_log = "Ollama API 請求超時"
                    status_code_for_log = 504
                    print(error_for_log)
                    yield f"data: {json.dumps({'error': error_for_log, 'status_code': status_code_for_log}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except aiohttp.ClientError as e:
                    error_for_log = f"Ollama API 連線錯誤: {str(e)}"
                    status_code_for_log = 503
                    print(error_for_log)
                    yield f"data: {json.dumps({'error': error_for_log, 'status_code': status_code_for_log}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_for_log = f"Ollama API 內部串流錯誤: {str(e)}"
                    status_code_for_log = 500
                    print(error_for_log)
                    yield f"data: {json.dumps({'error': error_for_log, 'status_code': status_code_for_log}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    completion_tokens_for_log = len(full_assistant_response_for_log)
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream; charset=utf-8"}
            )
        except Exception as e:
            status_code_for_log = 500
            error_for_log = f"Ollama API 錯誤 (外部): {str(e)}"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
    finally:
        if api_key_for_logging:
            log_api_usage(
                api_key=api_key_for_logging,
                request_type="chat_stream_completed",
                model_name=model_name_for_log,
                prompt_tokens=prompt_tokens_for_log,
                completion_tokens=completion_tokens_for_log,
                total_tokens=prompt_tokens_for_log + completion_tokens_for_log,
                input_summary=input_summary_for_logging,
                output_summary=full_assistant_response_for_log if full_assistant_response_for_log else None,
                status_code=status_code_for_log,
                error_message=error_for_log
            )

async def stream_chat_completion_minmax(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model
    full_assistant_response_for_log = ""
    prompt_tokens_for_log = len(input_summary_for_logging) if input_summary_for_logging else 0
    completion_tokens_for_log = 0
    error_for_log = None
    status_code_for_log = 200
    try:
        if not MINMAX_API_KEY: 
            status_code_for_log = 500
            error_for_log = "未設定 MinMax API 金鑰"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
        if not MINMAX_API_URL:
            status_code_for_log = 500
            error_for_log = "未設定 MinMax API URL"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)

        model_name = request.model.replace("minmax/", "")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINMAX_API_KEY.strip()}"
        }
        payload_data = {
            "model": model_name,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens is not None:
            payload_data["max_tokens"] = request.max_tokens

        async def generate_minmax_stream_content():
            nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
            server_sent_done = False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        MINMAX_API_URL, 
                        headers=headers,
                        json=payload_data,
                        timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT) 
                    ) as response:
                        status_code_for_log = response.status
                        if response.status != 200:
                            error_text = await response.text()
                            error_for_log = f"MinMax API 請求失敗 ({response.status}): {error_text}"
                            return

                        any_line_processed = False
                        async for line_bytes in response.content:
                            any_line_processed = True
                            line_bytes = line_bytes.strip()
                            if not line_bytes: continue
                            
                            line_str = ""
                            try:
                                line_str = line_bytes.decode('utf-8').strip()
                                if line_str.startswith("data:"):
                                    json_payload_str = line_str[len("data:"):].strip()
                                    if json_payload_str == "[DONE]": server_sent_done = True; break
                                    if not json_payload_str: continue
                                    try:
                                        chunk_data = json.loads(json_payload_str)
                                        mm_choices = chunk_data.get("choices")
                                        mm_content = None
                                        if mm_choices and len(mm_choices) > 0:
                                            if "delta" in mm_choices[0] and isinstance(mm_choices[0]["delta"], dict):
                                                mm_content = mm_choices[0]["delta"].get("content")
                                            elif "text" in mm_choices[0]: mm_content = mm_choices[0]["text"]
                                        if mm_content is not None:
                                            full_assistant_response_for_log += mm_content
                                            response_data = {'choices': [{'delta': {'content': mm_content}}]}
                                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                    except json.JSONDecodeError as e_json: print(f"MinMax stream: JSON parsing failed for 'data: ': '{json_payload_str}'. Error: {e_json}"); continue
                                else: # Direct JSON line
                                    try:
                                        chunk_data = json.loads(line_str)
                                        mm_choices = chunk_data.get("choices")
                                        mm_content = None
                                        if mm_choices and len(mm_choices) > 0:
                                            if "delta" in mm_choices[0] and isinstance(mm_choices[0]["delta"], dict):
                                                mm_content = mm_choices[0]["delta"].get("content")
                                            elif "text" in mm_choices[0]: mm_content = mm_choices[0]["text"]
                                        if mm_content is not None:
                                            full_assistant_response_for_log += mm_content
                                            response_data = {'choices': [{'delta': {'content': mm_content}}]}
                                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                    except json.JSONDecodeError: print(f"MinMax stream: Received non-SSE-data line, and also not valid JSON: '{line_str}'")
                            except UnicodeDecodeError as e_decode: print(f"MinMax stream: UTF-8 decoding failed: {e_decode}"); continue
                            except Exception as e_line_proc: print(f"MinMax stream: Error processing line '{line_str}': {e_line_proc}"); continue
                        if not any_line_processed: 
                            error_for_log = "MinMax stream: No lines were processed."
                            if status_code_for_log == 200 : status_code_for_log = 500 # Assume error
            except aiohttp.ClientError as e_client: error_for_log = f"MinMax stream (aiohttp.ClientError): {e_client}"; status_code_for_log = 503
            except asyncio.TimeoutError: error_for_log = "MinMax stream (TimeoutError)"; status_code_for_log = 504
            except Exception as e_general: error_for_log = f"MinMax stream (general error): {e_general}"; status_code_for_log = 500
            finally:
                completion_tokens_for_log = len(full_assistant_response_for_log)
                if not server_sent_done: yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_minmax_stream_content(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
    except Exception as e_stream_resp: # For errors creating StreamingResponse or pre-generator setup
        status_code_for_log = 500
        error_for_log = f"MinMax API 串流設定錯誤: {str(e_stream_resp)}"
        # This error is critical and happens before stream generator can be called or its finally block
        # So, we log it here immediately if api_key_for_logging is available
        if api_key_for_logging:
             log_api_usage(api_key_for_logging, "chat_stream_error", model_name_for_log, prompt_tokens_for_log, 0, prompt_tokens_for_log, input_summary_for_logging, None, status_code_for_log, error_for_log)
        raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
    finally:
        # This finally block is for the outer try/except of stream_chat_completion_minmax itself.
        # The generator's finally block handles logging for a successful or partially successful stream.
        # If an error occurred *before* the generator was even entered (e.g., in payload setup, or StreamingResponse creation failed),
        # this block might catch it. However, critical errors often get logged before StreamingResponse returns.
        # We need to ensure we don't double-log if the generator already logged.
        # For simplicity, if error_for_log is set by the main try-block (meaning generator likely didn't run or failed early),
        # we log it here. Otherwise, assume generator's finally logged.
        # This specific finally is tricky because the generator runs separately.
        # The current logic inside generate_minmax_stream_content's finally is the primary log point for stream content.
        pass # Primary logging for stream content is within the generator's finally. This outer finally is a safeguard.

async def stream_chat_completion_claude(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model
    full_assistant_response_for_log = ""
    prompt_tokens_for_log = len(input_summary_for_logging) if input_summary_for_logging else 0
    completion_tokens_for_log = 0
    error_for_log = None
    status_code_for_log = 200
    try:
        if not ANTHROPIC_API_KEY:
            status_code_for_log = 500
            error_for_log = "未設定 Anthropic API 金鑰"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)

        model_name = request.model.replace("claude/", "")
        system_prompt = None
        user_messages = []
        for msg in request.messages:
            if msg.role == "system": system_prompt = msg.content
            else: user_messages.append({"role": msg.role, "content": msg.content})
        if not user_messages: 
            status_code_for_log = 400; error_for_log = "Claude API 請求至少需要一條 user/assistant 訊息。"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)

        async def generate_claude_stream_content():
            nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
            server_sent_event_count = 0
            try:
                payload = {
                    "model": model_name,
                    "messages": user_messages,
                    "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
                    "temperature": request.temperature if request.temperature is not None else 0.7,
                    "stream": True
                }
                if system_prompt: payload["system"] = system_prompt
                headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
                url = "https://api.anthropic.com/v1/messages"
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        status_code_for_log = response.status
                        if response.status != 200:
                            error_text = await response.text()
                            error_for_log = f"Claude API 請求失敗 ({response.status}): {error_text}"
                            return
                        
                        async for line in response.content:
                            if not line.strip(): continue
                            line_str = line.decode('utf-8').strip()
                            if not line_str.startswith('data: '): continue
                            data_json = line_str[6:]
                            try:
                                data = json.loads(data_json)
                                event_type = data.get('type')
                                if event_type == 'content_block_delta':
                                    delta = data.get('delta', {})
                                    if delta.get('type') == 'text_delta' and 'text' in delta:
                                        content_delta = delta['text']
                                        full_assistant_response_for_log += content_delta
                                        response_data = {'choices': [{'delta': {'content': content_delta}}]}
                                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                                        server_sent_event_count += 1
                                elif event_type == 'message_stop': break
                            except json.JSONDecodeError: print(f"Claude stream: Error parsing JSON: {line_str}"); continue
            except Exception as e:
                error_for_log = f"Anthropic Claude API (stream) 錯誤: {str(e)}"
                status_code_for_log = 500 # Or parse from e
                print(error_for_log) # Added print for server log
                try: 
                    yield f"data: {json.dumps({'error': {'message': error_for_log, 'type': 'claude_stream_error'}})}\n\n"
                except Exception as yield_e: # Catch potential errors during yield itself
                    print(f"Error yielding Claude error message: {yield_e}")
            finally:
                completion_tokens_for_log = len(full_assistant_response_for_log)
        
        return StreamingResponse(
            generate_claude_stream_content(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"}
        )
    except Exception as e: # For errors creating StreamingResponse or pre-generator setup
        status_code_for_log = 500
        error_for_log = f"Claude API 串流設定錯誤: {str(e)}"
        if api_key_for_logging: # Log error if setup failed before stream starts
            log_api_usage(api_key_for_logging, "chat_stream_error", model_name_for_log, prompt_tokens_for_log, 0, prompt_tokens_for_log, input_summary_for_logging, None, status_code_for_log, error_for_log)
        raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
    finally:
        # The primary logging for stream content/errors is handled within the generator's finally block.
        # This outer finally is mainly a safeguard for errors that occur *before* the generator starts
        # or if the StreamingResponse object itself fails to be created.
        # If error_for_log is set here, it implies a setup issue before the stream truly began.
        if api_key_for_logging and error_for_log and completion_tokens_for_log == 0:
             log_api_usage(api_key_for_logging, "chat_stream_error", model_name_for_log, 
                           prompt_tokens_for_log, 0, prompt_tokens_for_log, 
                           input_summary_for_logging, None, 
                           status_code_for_log, error_for_log)
        # Else, assume the generator's finally block has handled or will handle logging.

async def stream_chat_completion_huggingface(request: ChatCompletionRequest, api_key_for_logging: Optional[str], input_summary_for_logging: Optional[str]):
    model_name_for_log = request.model
    full_assistant_response_for_log = ""
    prompt_tokens_for_log = len(input_summary_for_logging) if input_summary_for_logging else 0
    completion_tokens_for_log = 0
    error_for_log = None
    status_code_for_log = 200

    try:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
            from threading import Thread
        except ImportError:
            status_code_for_log = 500
            error_for_log = "伺服器錯誤：缺少 transformers 或 torch 庫"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
        
        model_name = request.model.replace('huggingface/', '', 1)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        except Exception as e:
            status_code_for_log = 500
            error_for_log = f"加載 Hugging Face 模型 {model_name} 失敗: {str(e)}"
            raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
        
        system_content = None; prompt = ""
        for msg in request.messages:
            if msg.role == "system": system_content = msg.content
            elif msg.role == "user": prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant": prompt += f"Assistant: {msg.content}\n"
        if system_content: prompt = f"System: {system_content}\n" + prompt
        prompt += "Assistant: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Approximate prompt tokens for HF based on tokenized input length, not just summary
        # This is a more accurate prompt token count for HF models if available.
        # Recalculate prompt_tokens_for_log based on actual tokenized input if different from summary length.
        actual_hf_prompt_tokens = len(inputs.input_ids[0])
        # Update prompt_tokens_for_log to be more accurate if desired, or keep as summary length.
        # For now, we'll stick to the consistent summary length for simplicity across all models as per user req.
        # prompt_tokens_for_log = actual_hf_prompt_tokens 

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=OLLAMA_TIMEOUT)
        gen_kwargs = dict(**inputs, max_new_tokens=request.max_tokens if request.max_tokens else 1024, temperature=request.temperature if request.temperature else 0.7, do_sample=True if request.temperature and request.temperature > 0 else False, streamer=streamer, pad_token_id=tokenizer.eos_token_id)
        
        thread = Thread(target=lambda: model.generate(**gen_kwargs))
        thread.start()
        
        async def generate():
            nonlocal full_assistant_response_for_log, completion_tokens_for_log, error_for_log, status_code_for_log
            try:
                for new_text in streamer:
                    if new_text:
                        full_assistant_response_for_log += new_text
                        response_data = {'choices': [{'delta': {'content': new_text}}]}
                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_for_log = f"Hugging Face 串流處理錯誤: {str(e)}"
                status_code_for_log = 500
                yield f"data: {json.dumps({'error': error_for_log, 'status_code': status_code_for_log}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                completion_tokens_for_log = len(full_assistant_response_for_log)
        
        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Content-Type": "text/event-stream; charset=utf-8"})
    except Exception as e: # For errors creating StreamingResponse or pre-generator setup
        status_code_for_log = 500
        error_for_log = f"Hugging Face 串流設定錯誤: {str(e)}"
        if api_key_for_logging:
            log_api_usage(api_key_for_logging, "chat_stream_error", model_name_for_log, prompt_tokens_for_log, 0, prompt_tokens_for_log, input_summary_for_logging, None, status_code_for_log, error_for_log)
        raise HTTPException(status_code=status_code_for_log, detail=error_for_log)
    finally:
        # Similar to Claude, the generator's finally block is primary for stream content log.
        # This outer finally is a safeguard.
        if api_key_for_logging and error_for_log and completion_tokens_for_log == 0:
            log_api_usage(api_key_for_logging, "chat_stream_error", model_name_for_log, 
                          prompt_tokens_for_log, 0, prompt_tokens_for_log, 
                          input_summary_for_logging, None, 
                          status_code_for_log, error_for_log)
        # Else, assume generator's finally has/will log.

# --- API 端點 ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request_data: ChatCompletionRequest, fastapi_request: Request, _authorized: bool = api_key_dependency):
    """處理聊天完成請求"""
    api_key_for_logging = await get_api_key_from_request(fastapi_request)
    input_summary_for_logging = request_data.messages[-1].content if request_data.messages else "No input"

    try:
        # 根據模型名稱和 stream 參數選擇適當的處理函數
        if request_data.model.startswith("openai/"):
            if request_data.stream:
                # For streaming, logging of full response and tokens happens inside stream_chat_completion_openai
                # Here we log the initial request attempt
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_openai(request_data, api_key_for_logging, input_summary_for_logging)
            else: # OpenAI non-stream
                response = await get_chat_completion_openai(request_data)
                if api_key_for_logging and response:
                    output_summary_val = None
                    prompt_tokens_val = 0
                    completion_tokens_val = 0
                    total_tokens_val = 0

                    if (response.choices and 
                        isinstance(response.choices[0], dict) and 
                        response.choices[0].get('message') and 
                        isinstance(response.choices[0]['message'], dict)):
                        output_summary_val = response.choices[0]['message'].get('content')
                    
                    if isinstance(response.usage, dict): # response.usage from ChatCompletionResponse is Dict[str, int]
                        prompt_tokens_val = response.usage.get("prompt_tokens", 0)
                        completion_tokens_val = response.usage.get("completion_tokens", 0)
                        total_tokens_val = response.usage.get("total_tokens", 0)
                    
                    log_api_usage(
                        api_key=api_key_for_logging,
                        request_type="chat",
                        model_name=response.model, # This is from ChatCompletionResponse.model
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        total_tokens=total_tokens_val,
                        input_summary=input_summary_for_logging,
                        output_summary=output_summary_val
                    )
                return response
        elif request_data.model.startswith("gemini/"):
            if request_data.stream:
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_gemini(request_data, api_key_for_logging, input_summary_for_logging) # Pass logging info
            else: # Gemini non-stream
                response = await get_chat_completion_gemini(request_data)
                if api_key_for_logging and response:
                    output_summary_val = None
                    prompt_tokens_val = 0 # Gemini non-stream usage is typically 0 from current get_chat_completion_gemini
                    completion_tokens_val = 0
                    total_tokens_val = 0

                    if (response.choices and 
                        isinstance(response.choices[0], dict) and 
                        response.choices[0].get('message') and 
                        isinstance(response.choices[0]['message'], dict)):
                        output_summary_val = response.choices[0]['message'].get('content')
                    
                    # get_chat_completion_gemini currently returns usage with all zeros
                    if isinstance(response.usage, dict):
                        prompt_tokens_val = response.usage.get("prompt_tokens", 0)
                        completion_tokens_val = response.usage.get("completion_tokens", 0)
                        total_tokens_val = response.usage.get("total_tokens", 0)

                    log_api_usage(
                        api_key=api_key_for_logging,
                        request_type="chat",
                        model_name=response.model,
                        prompt_tokens=prompt_tokens_val, # Will be 0 based on current get_chat_completion_gemini
                        completion_tokens=completion_tokens_val, # Will be 0
                        total_tokens=total_tokens_val, # Will be 0
                        input_summary=input_summary_for_logging,
                        output_summary=output_summary_val
                    )
                return response
        elif request_data.model.startswith("ollama/"):
            if request_data.stream:
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_ollama(request_data, api_key_for_logging, input_summary_for_logging)
            else:
                response = await get_chat_completion_ollama(request_data)
                # Ollama non-stream typically doesn't return token usage
                if api_key_for_logging and response:
                    output_summary_val = None
                    prompt_tokens_val = 0
                    completion_tokens_val = 0
                    total_tokens_val = 0

                    if (response.choices and 
                        isinstance(response.choices[0], dict) and 
                        response.choices[0].get('message') and 
                        isinstance(response.choices[0]['message'], dict)):
                        output_summary_val = response.choices[0]['message'].get('content')
                    
                    if isinstance(response.usage, dict):
                        prompt_tokens_val = response.usage.get("prompt_tokens", 0)
                        completion_tokens_val = response.usage.get("completion_tokens", 0)
                        total_tokens_val = response.usage.get("total_tokens", 0)
                        if total_tokens_val == 0 and (prompt_tokens_val > 0 or completion_tokens_val > 0):
                            total_tokens_val = prompt_tokens_val + completion_tokens_val
                    
                    log_api_usage(
                        api_key=api_key_for_logging,
                        request_type="chat",
                        model_name=response.model,
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        total_tokens=total_tokens_val,
                        input_summary=input_summary_for_logging,
                        output_summary=output_summary_val
                    )
                return response
        elif request_data.model.startswith("minmax/"):
            if request_data.stream:
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_minmax(request_data, api_key_for_logging, input_summary_for_logging)
            else:
                response = await get_chat_completion_minmax(request_data)
                if api_key_for_logging and response: log_api_usage(api_key_for_logging, "chat", response.model, prompt_tokens=response.usage.get("prompt_tokens",0), completion_tokens=response.usage.get("completion_tokens",0), total_tokens=response.usage.get("total_tokens",0), input_summary=input_summary_for_logging, output_summary=response.choices[0].message.content if response.choices else None)
                return response
        elif request_data.model.startswith("claude/"):
            if request_data.stream:
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_claude(request_data, api_key_for_logging, input_summary_for_logging)
            else:
                response = await get_chat_completion_claude(request_data)
                if api_key_for_logging and response: 
                    output_summary_val = None
                    prompt_tokens_val = 0
                    completion_tokens_val = 0
                    total_tokens_val = 0

                    if (response.choices and 
                        isinstance(response.choices[0], dict) and 
                        response.choices[0].get('message') and 
                        isinstance(response.choices[0]['message'], dict)):
                        output_summary_val = response.choices[0]['message'].get('content') # Corrected access for dict
                    else:
                        # Fallback or error if structure is not as expected, though get_chat_completion_claude should ensure it.
                        print(f"Warning: Claude non-stream response.choices structure unexpected: {response.choices}")
                    
                    if isinstance(response.usage, dict):
                        prompt_tokens_val = response.usage.get("prompt_tokens",0)
                        completion_tokens_val = response.usage.get("completion_tokens",0)
                        total_tokens_val = response.usage.get("total_tokens",0)
                    
                    log_api_usage(api_key_for_logging, "chat", response.model, 
                                  prompt_tokens=prompt_tokens_val, 
                                  completion_tokens=completion_tokens_val, 
                                  total_tokens=total_tokens_val, 
                                  input_summary=input_summary_for_logging, 
                                  output_summary=output_summary_val)
                return response
        elif request_data.model.startswith("huggingface/"):
            if request_data.stream:
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_huggingface(request_data, api_key_for_logging, input_summary_for_logging)
            else:
                response = await get_chat_completion_huggingface(request_data)
                if api_key_for_logging and response: log_api_usage(api_key_for_logging, "chat", response.model, prompt_tokens=response.usage.get("prompt_tokens",0), completion_tokens=response.usage.get("completion_tokens",0), total_tokens=response.usage.get("total_tokens",0), input_summary=input_summary_for_logging, output_summary=response.choices[0].message.content if response.choices else None)
                return response
        elif request_data.model.startswith("deepseek/"):
            if request_data.stream:
                # stream_chat_completion_openai handles deepseek streaming, need to pass logging info
                if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_stream_attempt", request_data.model, input_summary=input_summary_for_logging)
                return await stream_chat_completion_openai(request_data, api_key_for_logging, input_summary_for_logging) # Pass logging info
            else:
                response = await get_chat_completion_deepseek(request_data)
                if api_key_for_logging and response: log_api_usage(api_key_for_logging, "chat", response.model, prompt_tokens=response.usage.get("prompt_tokens",0), completion_tokens=response.usage.get("completion_tokens",0), total_tokens=response.usage.get("total_tokens",0), input_summary=input_summary_for_logging, output_summary=response.choices[0].message.content if response.choices else None)
                return response
        else:
            exc = HTTPException(status_code=400, detail=f"不支援的模型: {request_data.model}")
            if api_key_for_logging: log_api_usage(api_key_for_logging, "chat", request_data.model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=exc.detail)
            raise exc
    except HTTPException as e:
        # Log HTTP exceptions that occur during request processing
        if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_error", request_data.model, input_summary=input_summary_for_logging, status_code=e.status_code, error_message=e.detail)
        raise e
    except Exception as e:
        print(f"處理請求時發生錯誤: {str(e)}")  # 加入錯誤日誌
        exc = HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")
        if api_key_for_logging: log_api_usage(api_key_for_logging, "chat_error", request_data.model, input_summary=input_summary_for_logging, status_code=exc.status_code, error_message=str(e))
        raise exc

# --- 啟動伺服器 (用於本地測試) ---
if __name__ == "__main__":
    import uvicorn
    print("第一次安裝設定並啟動時，會等待比較久。請耐心等待直到 Application startup complete. 出現後，即可開始使用。")
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
    """處理 Google Gemini 的非串流聊天完成請求 (使用字元長度估算 token, 相容舊版 SDK)"""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="未設定 Google API 金鑰")
    if client is None: # client 是 genai.Client() 的全域實例
        raise HTTPException(status_code=500, detail="Google GenAI 客戶端未初始化。")

    model_name_for_api = request.model.replace('gemini/', '')
    if not model_name_for_api.startswith('models/'):
        model_name_for_api = f"models/{model_name_for_api}"
    
    print(f"使用 Gemini 模型 (非串流): {model_name_for_api}")
    
    system_instruction_text = None
    processed_messages = []
    prompt_char_count = 0

    for msg in request.messages:
        prompt_char_count += len(msg.content)
        if msg.role == "system":
            if system_instruction_text is None:
                system_instruction_text = msg.content
            else:
                processed_messages.append({"role": "user", "parts": [{"text": f"(額外系統指示): {msg.content}"}]})
        else:
            api_role = "user" if msg.role == "user" else "model"
            processed_messages.append({"role": api_role, "parts": [{"text": msg.content}]})
    
    final_contents_for_api = []
    if system_instruction_text:
        final_contents_for_api.append({"role": "user", "parts": [{"text": f"System Instruction: {system_instruction_text}\\n\\nUser Question/Input Below:"}]})
    final_contents_for_api.extend(processed_messages)

    if not final_contents_for_api or not any(part["role"] == "user" for part in final_contents_for_api):
        # 確保至少有一條 user message，這是 Gemini API 的普遍要求
        final_contents_for_api.append({"role": "user", "parts": [{"text": "(無用戶輸入，繼續執行)"}]})
        print("警告 (Gemini 非串流): 處理後 messages 列表為空或無 user message，已添加預設 user message。")

    api_params = {
        "model": model_name_for_api,
        "contents": final_contents_for_api,
    }
    # 移除 temperature 和 max_output_tokens
    # if request.temperature is not None:
    #     api_params["temperature"] = request.temperature
    # if request.max_tokens is not None:
    #     api_params["max_output_tokens"] = request.max_tokens

    print(f"Gemini 非串流 API 參數: {api_params}")

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            **api_params
        )
            
        completion_text = getattr(response, 'text', '') # 安全地獲取 text 屬性
        if not completion_text and response.candidates:
            # 有些情況下 text 可能為空，但 candidates[0].content.parts[0].text 有內容
            try:
                completion_text = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                completion_text = "[無法從 API 回應中提取內容]"

        completion_char_count = len(completion_text)
        total_char_count = prompt_char_count + completion_char_count
            
        return ChatCompletionResponse(
            id=f"gemini-ns-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop" 
            }],
            usage={ 
                "prompt_tokens": prompt_char_count, 
                "completion_tokens": completion_char_count,
                "total_tokens": total_char_count
            }
        )
    except Exception as e: 
        error_message = str(e)
        print(f"Gemini API 處理非串流請求時發生錯誤 ({model_name_for_api}): {error_message}")
        status_code = 500
        detail = f"Gemini API 錯誤 (非串流, 模型 {model_name_for_api}): {error_message}"

        if "API key not valid" in error_message or "Unauthenticated" in error_message: status_code = 401; detail = "Google API 金鑰無效或未認證。"
        elif "PermissionDenied" in error_message or "permission denied" in error_message: status_code = 403; detail = f"Google API 權限不足 ({model_name_for_api})。"
        elif "not found" in error_message.lower() and ("model" in error_message.lower() or "resource" in error_message.lower()): status_code = 404; detail = f"Gemini 模型 '{model_name_for_api}' 未找到。"
        elif "INVALID_ARGUMENT" in error_message or "Invalid request" in error_message or "got an unexpected keyword argument" in error_message : status_code = 400; detail = f"Gemini API 請求參數無效 ({model_name_for_api}): {error_message}"
        elif "blocked" in error_message.lower() and ("prompt" in error_message.lower() or "safety" in error_message.lower()): status_code = 400; detail = f"請求內容可能因安全政策被阻擋 ({model_name_for_api})。"
        elif "Deadline Exceeded" in error_message or "DEADLINE_EXCEEDED" in error_message: status_code = 504; detail = f"Gemini API 請求超時 ({model_name_for_api})。"
        
        if "module 'google.genai.types' has no attribute" in error_message: detail += " (SDK 版本問題，建議更新： pip install --upgrade google-generativeai)"
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
            api_url = OLLAMA_URL
            if not (api_url.endswith('/api/chat') or api_url.endswith('/api/chat/')):
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
                    print(f"Ollama API Error ({response.status}): {error_msg}") 
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Ollama API 回應錯誤: {error_msg}"
                    )
                
                result = await response.json()
                print(f"DEBUG: Ollama non-stream raw response: {result}")

                response_content = ""
                # 安全地獲取內容
                if (isinstance(result, dict) and 
                    "message" in result and
                    isinstance(result.get("message"), dict) and
                    "content" in result.get("message", {})):
                    response_content = result["message"]["content"]
                else:
                    print(f"錯誤：Ollama 非串流回應格式非預期。收到的 result: {result}")
                    response_content = "[Ollama 回應格式錯誤，無法提取內容]"
                    # 您也可以選擇在這裡拋出一個 HTTPException，如果這是不可接受的錯誤
                    # raise HTTPException(status_code=500, detail=f"Ollama 回應格式錯誤。原始回應: {str(result)[:200]}")

                # 嘗試獲取 token 使用量
                prompt_tokens = 0
                completion_tokens = 0
                actual_role = "assistant" # Default role

                if isinstance(result, dict):
                    prompt_tokens = result.get("prompt_eval_count", 0)
                    completion_tokens = result.get("eval_count", 0)
                    if isinstance(result.get("message"), dict):
                        actual_role = result["message"].get("role", "assistant")

                total_tokens = prompt_tokens + completion_tokens

                return ChatCompletionResponse(
                    id=f"ollama-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "message": {
                            "role": actual_role, 
                            "content": response_content
                        },
                        "finish_reason": "stop" 
                    }],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                )
    except Exception as e:
        import traceback
        print(f"Ollama API 錯誤 (get_chat_completion_ollama): {str(e)}")
        traceback.print_exc()
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
        
        print(f"Claude non-stream: Sending request to Anthropic API. System prompt: '{system_prompt if system_prompt else 'None'}'. Messages: {user_messages}")

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

# --- API 金鑰驗證依賴動態切換 ---
from fastapi import Depends, Header, HTTPException

def always_allow():
    return True

def verify_api_key(authorization: str = Header(None)):
    whitelist = os.getenv("api_keys_whitelist", "")
    whitelist_set = {k.strip() for k in whitelist.split(",") if k.strip()}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少或格式錯誤的 API 金鑰")
    token = authorization.replace("Bearer ", "").strip()
    if token not in whitelist_set:
        raise HTTPException(status_code=401, detail="API 金鑰無效")
    return True

ENABLE_CHECK_APIKEY = os.getenv("ENABLE_CHECK_APIKEY", "False").lower() == "true"
api_key_dependency = Depends(verify_api_key) if ENABLE_CHECK_APIKEY else Depends(always_allow)