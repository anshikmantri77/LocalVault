"""HuggingFace Inference API client (Mock/Fallback)."""

# This dict is used by rag_pipeline.py and streamlit_app.py
AVAILABLE_MODELS = {
    "GLM-4-9B": "THUDM/glm-4-9b-chat",
    "MiniCPM-o (Kimi)": "openbmb/MiniCPM-o-2_6",
    "Phi-4 Mini": "microsoft/phi-4-mini",
    "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen 2.5 7B (Fast)": "Qwen/Qwen2.5-7B-Instruct",
}

def get_hf_response(prompt: str, model_id: str):
    """
    Placeholder for HuggingFace Inference API call.
    In a real scenario, this would use a token and call the HF API.
    Since LocalVault focuses on local Ollama, this is a placeholder.
    """
    return {
        "content": f"HF Model {model_id} is currently in 'Cloud-Only' mode. Please use the local Ollama model for 100% privacy.",
        "success": False
    }
