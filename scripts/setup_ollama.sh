#!/bin/bash

echo "🚀 Setting up Ollama for local LLM serving..."

# Install Ollama (macOS/Linux)
if command -v ollama &> /dev/null; then
    echo "✅ Ollama already installed"
else
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &
sleep 5

# Download base model
echo "📦 Downloading Llama3 8B model..."
ollama pull llama3:8b

# Verify installation
echo "✅ Verifying Ollama installation..."
if ollama list | grep -q "llama3:8b"; then
    echo "🎉 Ollama setup complete! Model ready for use."
else
    echo "❌ Setup failed. Please check Ollama installation."
fi
