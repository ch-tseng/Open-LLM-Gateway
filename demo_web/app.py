import streamlit as st
import requests
import json
import os
import sseclient
import configparser
from pathlib import Path

# 設定頁面配置和標題 - 必須是第一個Streamlit命令
st.set_page_config(
    page_title="多模型LLM聊天界面",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 默認模型設定 - 硬編碼所有模型
DEFAULT_MODELS = {
    "openai": [ "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4", "o3", "o3-mini", "o1", "o1-pro"],
    "claude": ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06", "gemini-1.0-pro"],
    "deepseek": ["deepseek-chat"],
    "minmax": ["minimax-text-01", "abab6.5s-chat"],
    "ollama": ["qwen3:8b", "gemma3:12b"],
    "huggingface": ["meta-llama/Llama-3-8b-chat", "meta-llama/Llama-3-70b-chat", "mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-7b-it", "microsoft/phi-3-medium-4k-instruct"]
}

DEFAULT_API_ENDPOINT = "http://172.30.11.15:8000"
API_KEY = "AT0130-20250508-kbxk8c"

# 載入配置 - 簡化為直接使用默認值
MODELS = DEFAULT_MODELS
DEFAULT_ENDPOINT = DEFAULT_API_ENDPOINT

# 嘗試從models.conf載入設定（如果存在）
config_file = Path(__file__).parent / "models.conf"
if config_file.exists():
    try:
        # 嘗試使用ConfigParser讀取配置（如果文件存在）
        config = configparser.ConfigParser()
        try:
            config.read(config_file, encoding='utf-8')
        except:
            pass  # 忽略讀取錯誤，使用默認值
        
        # 如果成功讀取了general區段，嘗試獲取API端點
        if 'general' in config and 'default_api_endpoint' in config['general']:
            DEFAULT_ENDPOINT = config['general']['default_api_endpoint']
            
        st.sidebar.success("成功載入API端點設定")
    except Exception:
        pass  # 忽略任何錯誤，使用默認值
else:
    st.sidebar.info("未找到models.conf配置文件，使用默認設定")

# 初始化會話狀態
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_endpoint' not in st.session_state:
    st.session_state.api_endpoint = DEFAULT_ENDPOINT

if 'api_key' not in st.session_state:
    st.session_state.api_key = API_KEY

if 'selected_vendor' not in st.session_state:
    st.session_state.selected_vendor = "openai"

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = MODELS["openai"][0]

# 設定側邊欄
with st.sidebar:
    st.title("🤖 多模型LLM聊天")
    
    # API端點設定
    st.subheader("API端點設定")
    api_endpoint = st.text_input(
        "LLM主機 (IP:port)",
        value=st.session_state.api_endpoint
    )
    
    # 更新API端點
    if api_endpoint != st.session_state.api_endpoint:
        st.session_state.api_endpoint = api_endpoint
    
    # API金鑰設定
    api_key = st.text_input(
        "API金鑰",
        value=st.session_state.api_key,
        type="password"
    )
    
    # 更新API金鑰
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # LLM供應商選擇
    st.subheader("LLM供應商選擇")
    vendor = st.selectbox(
        "選擇LLM供應商",
        options=["openai", "claude", "gemini", "deepseek", "minmax", "ollama", "huggingface"],
        index=["openai", "claude", "gemini", "deepseek", "minmax", "ollama", "huggingface"].index(st.session_state.selected_vendor)
    )
    
    # 更新供應商選擇
    if vendor != st.session_state.selected_vendor:
        st.session_state.selected_vendor = vendor
        # 當切換供應商時，選擇該供應商的第一個模型
        st.session_state.selected_model = MODELS[vendor][0]
    
    # 模型選擇（下拉選單）
    available_models = MODELS[vendor]
    if st.session_state.selected_model in available_models:
        model_index = available_models.index(st.session_state.selected_model)
    else:
        model_index = 0
    
    model = st.selectbox(
        f"選擇{vendor}模型",
        options=available_models,
        index=model_index
    )
    
    # 更新模型選擇
    if model != st.session_state.selected_model:
        st.session_state.selected_model = model
    
    # 串流選項
    stream_enabled = st.checkbox("啟用串流回應", value=True)
    
    # 清除對話選項
    if st.button("清除對話"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("© 2024 LLM網關聊天界面")

# 主要聊天界面
st.title("💬 聊天界面")

# 顯示聊天訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 函數：發送非串流請求
def send_non_stream_request(prompt):
    api_url = f"{st.session_state.api_endpoint}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.api_key}"
    }
    
    # 構建完整模型名稱（加上供應商前綴）
    full_model_name = f"{st.session_state.selected_vendor}/{st.session_state.selected_model}"
    
    # 構建請求數據
    data = {
        "model": full_model_name,
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        with st.spinner("AI思考中..."):
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=90)
            response.raise_for_status()
            result = response.json()
            
            if result.get('choices') and len(result['choices']) > 0:
                assistant_message = result['choices'][0]['message']['content']
                return assistant_message
            else:
                return "無法獲取有效回應。請檢查API設定或稍後再試。"
    
    except requests.exceptions.RequestException as e:
        error_message = f"請求錯誤: {str(e)}"
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                return f"錯誤詳情: {json.dumps(error_detail)}"
            except:
                return f"錯誤狀態碼: {e.response.status_code}, 錯誤內容: {e.response.text}"
        return error_message

# 函數：發送串流請求
def send_stream_request(prompt):
    api_url = f"{st.session_state.api_endpoint}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.api_key}"
    }
    
    # 構建完整模型名稱（加上供應商前綴）
    full_model_name = f"{st.session_state.selected_vendor}/{st.session_state.selected_model}"
    
    # 構建請求數據
    data = {
        "model": full_model_name,
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        "temperature": 0.7,
        "stream": True
    }
    
    try:
        # 創建一個空的佔位回應
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with requests.post(api_url, headers=headers, data=json.dumps(data), stream=True) as response:
                response.raise_for_status()
                client = sseclient.SSEClient(response)
                
                for event in client.events():
                    if event.data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(event.data)
                        if chunk.get('choices') and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                full_response += delta['content']
                                message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
                
                # 最終顯示完整回應（無閃爍游標）
                message_placeholder.markdown(full_response)
                return full_response
    
    except requests.exceptions.RequestException as e:
        error_message = f"請求錯誤: {str(e)}"
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                return f"錯誤詳情: {json.dumps(error_detail)}"
            except:
                return f"錯誤狀態碼: {e.response.status_code}, 錯誤內容: {e.response.text}"
        return error_message

# 聊天輸入
if prompt := st.chat_input("輸入您的訊息..."):
    # 添加用戶訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 顯示用戶訊息
    with st.chat_message("user"):
        st.write(prompt)
    
    # 獲取並顯示助手回應
    if stream_enabled:
        assistant_response = send_stream_request(prompt)
    else:
        # 顯示助手訊息（非串流模式）
        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                assistant_response = send_non_stream_request(prompt)
                st.write(assistant_response)
    
    # 將助手訊息添加到會話歷史
    st.session_state.messages.append({"role": "assistant", "content": assistant_response}) 