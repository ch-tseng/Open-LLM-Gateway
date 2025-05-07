# 多模型LLM聊天界面

這是一個使用Streamlit構建的網頁應用程序，可連接到各種LLM供應商的API網關。

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