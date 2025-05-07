@echo off
echo 檢查依賴套件...
pip install -r requirements.txt

echo 啟動多模型LLM聊天界面...
streamlit run app.py

pause 