import streamlit as st
import os
import re
import json
import random
import string
import datetime
import pandas as pd
from pathlib import Path
import shutil
import configparser

# 設定頁面配置和標題
st.set_page_config(
    page_title="API金鑰管理頁面",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 常數定義
HISTORY_APIKEY_DIR = "../history_apikey"
API_KEYS_FILE = "api_keys.json"  # 存儲所有API金鑰信息的文件
DISABLED_MARK = "_disabled"  # 用於標記已停用的API金鑰

# 設定登入密碼（實際應用中應使用更安全的認證方式）
ADMIN_PASSWORD = "admin123"

# 確保API金鑰目錄存在
os.makedirs(HISTORY_APIKEY_DIR, exist_ok=True)

# 加載API金鑰數據
def load_api_keys():
    api_keys_path = os.path.join(HISTORY_APIKEY_DIR, API_KEYS_FILE)
    if os.path.exists(api_keys_path):
        with open(api_keys_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 保存API金鑰數據
def save_api_keys(api_keys):
    api_keys_path = os.path.join(HISTORY_APIKEY_DIR, API_KEYS_FILE)
    with open(api_keys_path, 'w', encoding='utf-8') as f:
        json.dump(api_keys, f, ensure_ascii=False, indent=4)

# 產生新的API金鑰
def generate_apikey(prefix):
    """產生格式為 {前綴}-{當前日期}-{6位隨機字符} 的API金鑰"""
    date_part = datetime.datetime.now().strftime("%Y%m%d")
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}-{date_part}-{random_part}"

# 驗證API金鑰格式
def validate_apikey_format(apikey):
    """驗證API金鑰格式"""
    pattern = r"^[A-Z0-9]{2,8}-\d{8}-[a-z0-9]{6}$"
    return bool(re.match(pattern, apikey))

# 獲取API金鑰使用統計
def get_apikey_stats():
    """從history_apikey目錄獲取所有API金鑰的使用統計"""
    api_keys_data = load_api_keys() # Changed variable name for clarity
    stats = []
    
    # 遍歷history_apikey目錄中的所有API金鑰目錄
    if not os.path.exists(HISTORY_APIKEY_DIR):
        print(f"Warning: API key history directory not found: {HISTORY_APIKEY_DIR}")
        return stats
        
    for apikey_dir_name in os.listdir(HISTORY_APIKEY_DIR):
        apikey_path = os.path.join(HISTORY_APIKEY_DIR, apikey_dir_name)
        
        # 跳過非目錄項或API金鑰數據文件
        if not os.path.isdir(apikey_path) or apikey_dir_name == API_KEYS_FILE:
            continue
        
        # 獲取API金鑰名稱 (原始目錄名，可能包含 _disabled)
        raw_apikey_name = apikey_dir_name
        
        is_disabled = raw_apikey_name.endswith(DISABLED_MARK)
        if is_disabled:
            actual_apikey = raw_apikey_name[:-len(DISABLED_MARK)]
        else:
            actual_apikey = raw_apikey_name

        # 檢查是否為有效的API金鑰格式 (針對 actual_apikey)
        if not validate_apikey_format(actual_apikey):
            print(f"Skipping directory with invalid API key format: {actual_apikey}")
            continue
            
        total_requests = 0
        tokens_in = 0
        tokens_out = 0
        last_access_dt = None # Use datetime object for comparison
        
        # 遍歷API金鑰目錄中的所有日誌文件 (e.g., YYYYMMDD.txt)
        for log_file_name in os.listdir(apikey_path):
            if not log_file_name.endswith('.txt'): # Assuming logs are .txt files
                continue
                
            log_file_path = os.path.join(apikey_path, log_file_name)
            
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        total_requests += 1 # Count each non-empty line as a request attempt

                        # Regex to capture relevant parts from the new log format
                        # Example: [2024-05-15 10:00:00] Type: chat, Model: openai/gpt-3.5, APIKeySuffix: ...key, PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30, StatusCode: 200
                        match = re.search(
                            r"""^\[(.*?)\]                        # Timestamp (e.g., 2024-05-15 10:00:00)
                            .*?Type:\s*(embedding|chat|chat_stream_attempt|chat_stream_completed|chat_error)  # Type
                            .*?Model:\s*(.*?)                      # Model Name
                            .*?APIKeySuffix:\s*(.*?)               # API Key Suffix
                            .*?PromptTokens:\s*(\d+)               # Prompt Tokens
                            .*?CompletionTokens:\s*(\d+)           # Completion Tokens
                            .*?TotalTokens:\s*(\d+)                 # Total Tokens
                            .*?StatusCode:\s*(\d+)                 # Status Code
                            (?:.*?Input:\s*(.*?))?                  # Optional Input Summary
                            (?:.*?Output:\s*(.*?))?                 # Optional Output Summary
                            (?:.*?Error:\s*(.*?))?$                 # Optional Error Message
                            """, 
                            line, 
                            re.VERBOSE
                        )

                        if match:
                            timestamp_str = match.group(1)
                            # request_type = match.group(2) # Not used for current stats, but available
                            # model_name_logged = match.group(3) # Not used for current stats
                            prompt_tokens_val = int(match.group(5))
                            completion_tokens_val = int(match.group(6))
                            # total_tokens_val = int(match.group(7)) # Can be used or re-calculated
                            status_code_val = int(match.group(8))

                            # Only add to token counts if request was successful (e.g., StatusCode 200)
                            # You might want to adjust this logic based on how errors are logged or if you want to count tokens for failed requests too.
                            if 200 <= status_code_val < 300: # Consider 2xx as success for token counting
                                tokens_in += prompt_tokens_val
                                tokens_out += completion_tokens_val
                            
                            try:
                                current_access_dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                                if last_access_dt is None or current_access_dt > last_access_dt:
                                    last_access_dt = current_access_dt
                            except ValueError:
                                print(f"Warning: Could not parse timestamp '{timestamp_str}' in {log_file_path}, line {line_number + 1}")
                        else:
                            # Fallback for old log format or lines that don't match
                            # This part tries to salvage data if old logs exist or for simpler log lines.
                            # It might be less accurate or conflict if new logs are partially formed.
                            # Consider removing or adjusting this fallback if only new log format is expected.
                            legacy_token_match = re.search(r'Tokens: (\d+)', line)
                            if legacy_token_match:
                                tokens_val = int(legacy_token_match.group(1))
                                tokens_in += tokens_val # Old format didn't distinguish in/out well
                                # tokens_out += int(tokens_val * 0.7) # Old approximation
                            
                            legacy_ts_match = re.search(r'\[(.*?)]', line)
                            if legacy_ts_match:
                                try:
                                    current_access_dt = datetime.datetime.strptime(legacy_ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                                    if last_access_dt is None or current_access_dt > last_access_dt:
                                        last_access_dt = current_access_dt
                                except ValueError:
                                    pass # Ignore if timestamp in old format is also unparseable
                            # print(f"Warning: Could not parse log line in {log_file_path}, line {line_number + 1}: {line[:100]}...")
            except Exception as e:
                print(f"Error reading or processing log file {log_file_path}: {e}")
                continue # Skip to next log file if one is broken
        
        # 獲取創建時間 (從API金鑰的日期部分，或從 api_keys_data)
        key_creation_info = api_keys_data.get(actual_apikey, {}).get("created_at")
        if key_creation_info:
            # Assuming created_at is stored as "YYYY-MM-DD HH:MM:SS"
            created_at_str = key_creation_info.split(' ')[0] # Get just the date part
        else:
            try:
                # Fallback to parsing from key name if not in api_keys.json (e.g. manually created folders)
                date_part_from_key = actual_apikey.split('-')[1]
                created_at_str = f"{date_part_from_key[:4]}-{date_part_from_key[4:6]}-{date_part_from_key[6:8]}"
            except IndexError:
                created_at_str = "未知"
        
        # 獲取描述（如果存在）
        description = api_keys_data.get(actual_apikey, {}).get("description", "")
        
        # 添加到統計列表
        stats.append({
            "api_key": actual_apikey,
            "created_at": created_at_str,
            "last_access": last_access_dt.strftime("%Y-%m-%d %H:%M:%S") if last_access_dt else "從未使用",
            "total_requests": total_requests,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "description": description,
            "is_disabled": is_disabled
        })
    
    return stats

# 刪除API金鑰
def delete_apikey(apikey):
    """刪除指定的API金鑰及其所有日誌"""
    # 刪除API金鑰目錄
    apikey_path = os.path.join(HISTORY_APIKEY_DIR, apikey)
    if os.path.exists(apikey_path):
        shutil.rmtree(apikey_path)
    
    # 從API金鑰配置中刪除
    api_keys = load_api_keys()
    if apikey in api_keys:
        del api_keys[apikey]
        save_api_keys(api_keys)
    
    return True

# 停用/啟用API金鑰
def toggle_apikey_status(apikey, is_disabled):
    """停用或啟用API金鑰"""
    api_keys = load_api_keys()
    
    # 原目錄路径
    orig_path = os.path.join(HISTORY_APIKEY_DIR, apikey)
    
    if is_disabled:
        # 停用API金鑰
        new_key = f"{apikey}{DISABLED_MARK}"
        new_path = os.path.join(HISTORY_APIKEY_DIR, new_key)
    else:
        # 啟用API金鑰
        new_key = apikey.replace(DISABLED_MARK, "")
        new_path = os.path.join(HISTORY_APIKEY_DIR, new_key)
    
    # 重命名目錄
    if os.path.exists(orig_path):
        os.rename(orig_path, new_path)
    
    # 更新配置
    if apikey in api_keys:
        api_keys[new_key] = api_keys[apikey]
        del api_keys[apikey]
    
    save_api_keys(api_keys)
    return True

# 添加API金鑰到白名單
def add_to_whitelist(apikey, description=""):
    """將API金鑰添加到白名單"""
    api_keys = load_api_keys()
    api_keys[apikey] = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description
    }
    save_api_keys(api_keys)
    
    # 創建API金鑰目錄
    apikey_path = os.path.join(HISTORY_APIKEY_DIR, apikey)
    os.makedirs(apikey_path, exist_ok=True)
    
    return True

# 更新API金鑰描述
def update_apikey_description(apikey, description):
    """更新API金鑰描述"""
    api_keys = load_api_keys()
    if apikey in api_keys:
        api_keys[apikey]["description"] = description
        save_api_keys(api_keys)
        return True
    return False

# 更新.env文件中的白名單
def update_env_whitelist():
    """更新主.env文件中的API金鑰白名單"""
    api_keys = load_api_keys()
    active_keys = [key for key in api_keys.keys() if not key.endswith(DISABLED_MARK)]
    
    env_path = os.path.join("..", ".env")
    if os.path.exists(env_path):
        # 讀取現有.env內容
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
        
        # 更新api_keys_whitelist項
        whitelist_line = f"api_keys_whitelist={','.join(active_keys)}"
        
        # 檢查是否已有api_keys_whitelist行
        if "api_keys_whitelist=" in env_content:
            # 替換現有行
            env_content = re.sub(r"api_keys_whitelist=.*", whitelist_line, env_content)
        else:
            # 添加新行
            env_content += f"\n{whitelist_line}"
        
        # 寫回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        return True
    return False

# 主應用程序
def main():
    st.title("🔑 API金鑰管理頁面")
    
    # 簡單的登錄系統
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("login_form"):
            password = st.text_input("請輸入管理員密碼", type="password")
            submit = st.form_submit_button("登入")
            
            if submit:
                if password == ADMIN_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("密碼錯誤")
        return
    
    # 主界面（登入後）
    tab1, tab2 = st.tabs(["API金鑰管理", "產生新API金鑰"])
    
    # 標籤1：API金鑰管理
    with tab1:
        st.header("API金鑰列表及使用統計")
        
        # 獲取API金鑰統計數據
        apikey_stats = get_apikey_stats()
        
        if not apikey_stats:
            st.info("尚未找到任何API金鑰記錄。請先產生新的API金鑰。")
        else:
            # 轉換為DataFrame以便顯示
            df = pd.DataFrame(apikey_stats)
            
            # 添加操作按鈕的占位列
            df['操作'] = None
            
            # 顯示表格
            st.dataframe(df[['api_key', 'description', 'created_at', 'last_access', 'total_requests', 'tokens_in', 'tokens_out', 'is_disabled']], hide_index=True)
            
            # 刪除和停用按鈕
            st.subheader("API金鑰操作")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_key = st.selectbox("選擇API金鑰", [item["api_key"] for item in apikey_stats])
            
            with col2:
                is_disabled = any(item["api_key"] == selected_key and item["is_disabled"] for item in apikey_stats)
                if is_disabled:
                    if st.button("啟用API金鑰"):
                        if toggle_apikey_status(selected_key, False):
                            st.success(f"已啟用API金鑰: {selected_key}")
                            update_env_whitelist()
                            st.rerun()
                else:
                    if st.button("停用API金鑰"):
                        if toggle_apikey_status(selected_key, True):
                            st.success(f"已停用API金鑰: {selected_key}")
                            update_env_whitelist()
                            st.rerun()
            
            with col3:
                if st.button("刪除API金鑰", type="primary", help="警告：此操作不可逆"):
                    if delete_apikey(selected_key):
                        st.success(f"已刪除API金鑰: {selected_key}")
                        update_env_whitelist()
                        st.rerun()
            
            # 更新描述
            st.subheader("更新API金鑰描述")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_key_info = next((item for item in apikey_stats if item["api_key"] == selected_key), None)
                current_description = selected_key_info["description"] if selected_key_info else ""
                new_description = st.text_input("描述", value=current_description)
            
            with col2:
                if st.button("更新描述"):
                    if update_apikey_description(selected_key, new_description):
                        st.success(f"已更新API金鑰描述")
                        st.rerun()
    
    # 標籤2：產生新API金鑰
    with tab2:
        st.header("產生新API金鑰")
        
        with st.form("generate_apikey_form"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                prefix = st.text_input("功能代碼 (2-8個字母或數字)", max_chars=8).upper()
            
            with col2:
                description = st.text_input("API金鑰描述 (用途、負責人等)")
            
            submit = st.form_submit_button("產生API金鑰")
            
            if submit:
                if not prefix:
                    st.error("請輸入功能代碼")
                elif not re.match(r"^[A-Z0-9]{2,8}$", prefix):
                    st.error("功能代碼必須是2-8個大寫字母或數字")
                else:
                    new_apikey = generate_apikey(prefix)
                    
                    # 添加到白名單
                    add_to_whitelist(new_apikey, description)
                    
                    # 更新.env文件
                    update_env_whitelist()
                    
                    st.success(f"已成功產生新API金鑰：{new_apikey}")
                    st.code(new_apikey)

if __name__ == "__main__":
    main() 