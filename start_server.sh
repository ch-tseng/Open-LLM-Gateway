echo "伺服器正在啟動，第一次執行或依賴較多時可能需要較長時間，請耐心等待直到看見 'Application startup complete.' ..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
