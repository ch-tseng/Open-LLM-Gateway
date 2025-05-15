#!/bin/bash

# 檢查是否安裝了依賴包
if [ ! -f "$(which streamlit)" ]; then
    echo "正在安裝依賴庫..."
    pip install -r requirements.txt
fi

# 啟動Streamlit應用
echo "啟動多模型LLM聊天界面..."
streamlit run app.py 