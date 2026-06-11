FROM python:3.11-slim

WORKDIR /workspace

# System dependencies for OpenCV + curl for Ollama install
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install CPU-only PyTorch FIRST using the official CPU wheel index.
# This avoids pip pulling the 500MB+ CUDA version that bert-score would trigger.
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install everything else.
# bert-score will now reuse the already-installed CPU torch instead of re-downloading.
COPY requirements_silva.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements_silva.txt

# Download CNN model files from HuggingFace
RUN mkdir -p /workspace/models && \
    curl -fsSL -o /workspace/models/binary_model.h5 \
        https://huggingface.co/MaryamRafaqat/breast-cancer-densenet121/resolve/main/binary_model.h5 && \
    curl -fsSL -o /workspace/models/benign_subtype_model.h5 \
        https://huggingface.co/MaryamRafaqat/breast-cancer-densenet121/resolve/main/benign_subtype_model.h5 && \
    curl -fsSL -o /workspace/models/malignant_subtype_model.h5 \
        https://huggingface.co/MaryamRafaqat/breast-cancer-densenet121/resolve/main/malignant_subtype_model.h5

# Copy all job scripts
COPY 00_setup_models/    /workspace/00_setup_models/
COPY 01_cnn_inference/   /workspace/01_cnn_inference/
COPY 02_llm_report/      /workspace/02_llm_report/
COPY 03_display_results/ /workspace/03_display_results/

RUN chmod +x /workspace/00_setup_models/run.sh \
             /workspace/01_cnn_inference/run.sh \
             /workspace/02_llm_report/run.sh \
             /workspace/03_display_results/run.sh

# Install Ollama and bake in the mistral model
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama serve & \
    until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done && \
    ollama pull mistral

CMD ["/bin/bash"]
