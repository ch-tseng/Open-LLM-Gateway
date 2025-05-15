# 多模型 LLM Gateway Web 界面

這是一個使用 Streamlit 構建的 Web 界面，用於與多模型 LLM 網關 API 進行互動。

## 主要功能

1. **聊天界面**：與各種 LLM 模型進行聊天
2. **API金鑰管理**：產生、管理和監控 API 金鑰使用情況

## 應用程序清單

- **app.py**：主要聊天界面
- **admin.py**：API金鑰管理界面

## 安裝和運行

1. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

2. **運行聊天界面**：
   ```bash
   streamlit run app.py
   ```

3. **運行管理界面**：
   ```bash
   streamlit run admin.py
   ```

## API金鑰管理頁面功能

管理頁面 (`admin.py`) 提供以下功能：

1. **產生API金鑰**：
   - 管理者輸入功能代碼 (前綴)
   - 系統自動生成格式為 `{功能代碼}-{日期}-{隨機字符}` 的API金鑰

2. **API金鑰使用統計**：
   - 顯示所有API金鑰的使用情況
   - 統計包括：最近訪問時間、訪問次數、輸入/輸出tokens數量

3. **API金鑰管理**：
   - 停用/啟用API金鑰
   - 刪除API金鑰
   - 更新API金鑰描述

4. **自動更新白名單**：
   - 自動將有效API金鑰添加到主配置的白名單中

## 安全提示

管理頁面使用簡單密碼保護，默認密碼為 `admin123`。在生產環境中，請務必修改默認密碼。

## 依賴項

- streamlit
- requests
- python-dotenv
- sseclient-py
- pandas

## 功能特點

- 支持多種LLM供應商：OpenAI、Claude、Google Gemini、DeepSeek、MinMax、Ollama
- 可輕鬆配置API端點（主機IP:port）
- 每個供應商支持多個模型選項
- 可選擇串流或非串流模式
- 簡潔直觀的聊天界面
- 保存聊天歷史

## 安裝步驟

1. 安裝依賴套件：

```bash
cd demo_web
pip install -r requirements.txt
```

2. （可選）修改配置文件：

應用使用 `models.conf` 配置文件，定義每個LLM供應商的可用模型列表：

```ini
# models.conf 範例
[openai]
models=gpt-3.5-turbo,gpt-4,gpt-4-turbo,gpt-4o

[claude]
models=claude-3-haiku-20240307,claude-3-sonnet-20240229

[general]
default_api_endpoint=http://localhost:8000
```

你可以根據需要新增或刪除每個供應商的模型選項。

## 啟動應用

```bash
cd demo_web
streamlit run app.py
```

或者使用提供的啟動腳本：

**Linux/Mac**:
```bash
cd demo_web
chmod +x start.sh
./start.sh
```

**Windows**:
```
cd demo_web
run.bat
```

應用將啟動並自動打開瀏覽器窗口，或者你可以手動訪問 `http://localhost:8501`。

## 使用方法

1. 在側邊欄輸入LLM API網關的主機地址（例如：`172.30.11.15:8000`）
2. 從下拉菜單選擇LLM供應商（OpenAI、Claude等）
3. 從下拉菜單中選擇該供應商的模型
4. 選擇是否啟用串流模式
5. 在底部輸入框輸入訊息並發送
6. 可隨時使用"清除對話"按鈕重新開始

## 注意事項

- 確保API網關服務正在運行且可訪問
- 串流模式提供更流暢的響應體驗，但可能消耗更多資源
- 對話歷史保存在瀏覽器會話中，重新啟動應用會清除歷史
- 如需添加新模型，只需編輯 `models.conf` 文件 
