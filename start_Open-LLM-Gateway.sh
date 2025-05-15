source ~/envs/openLLM/bin/activate
cd /GPUData/projs/Open-LLM-Gateway
uvicorn main:app --reload --host 0.0.0.0 --port 8000
