# 1. Create clean environment
conda create -n llm-trace python=3.10 -y
conda activate llm-trace

# 2. Install PyTorch with CUDA 12.6 support
pip3 install torch --index-url https://download.pytorch.org/whl/cu126

# 3. Install core packages with pinned versions
pip install \
  transformers==4.57.1 \
  accelerate==1.12.0 \
  tokenizers==0.22.1 \
  tqdm==4.66.4

# 4. Test with a small model
python main_test_2.py --model_name Qwen/Qwen3-0.6B-Base