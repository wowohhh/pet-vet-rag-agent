#!/bin/bash
# 启动宠物兽医知识助手
export OLLAMA_MODELS="C:/ollama_models"
export PYTHONPATH="$(dirname "$0")"
streamlit run "$(dirname "$0")/src/ui/app.py" --server.port 8501
