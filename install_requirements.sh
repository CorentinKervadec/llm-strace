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
python main.py --model_name Qwen/Qwen3-0.6B-Base

# You should obtain something like this

# ==================================================================================
# [s-Trace] COMPUTATIONAL DENSITY & RECONSTRUCTION SUMMARY
# ==================================================================================
# Rel. Size (s)       TV Error (Trace)    TV Error (Rand)     Stage / Phase         
# ----------------------------------------------------------------------------------
#  0.001% (8.2e-06)   0.9798              0.9983              Construction Phase    
#  0.010% (9.9e-05)   0.9798              0.9990              Construction Phase    
#  0.020% (2.0e-04)   0.9798              0.9989              Construction Phase    
#  0.040% (4.0e-04)   0.9798              0.9985              Construction Phase    
#  0.080% (8.0e-04)   0.9896              0.9997              Construction Phase    
#  0.100% (1.0e-03)   0.9884              0.9963              ★ Minimal Core (~10⁻³)
#  0.119% (1.2e-03)   0.9884              0.9980              ★ Minimal Core (~10⁻³)
#  0.140% (1.4e-03)   0.9892              0.9996              ★ Minimal Core (~10⁻³)
#  0.199% (2.0e-03)   0.9906              0.9998              Construction Phase    
#  0.300% (3.0e-03)   0.9808              0.9990              Construction Phase    
#  0.399% (4.0e-03)   0.9681              0.9996              Construction Phase    
#  0.599% (6.0e-03)   0.9719              0.9986              Construction Phase    
#  0.799% (8.0e-03)   0.9766              0.9967              Construction Phase    
#  0.999% (1.0e-02)   0.9883              0.9906              Construction Phase    
#  2.000% (2.0e-02)   0.9844              0.9912              Refinement Phase      
#  3.999% (4.0e-02)   0.9835              0.9945              Refinement Phase      
#  5.999% (6.0e-02)   0.1187              0.9942              Refinement Phase      
#  8.000% (8.0e-02)   0.0810              0.9485              Refinement Phase      
# 10.000% (1.0e-01)   0.0238              0.9083              Refinement Phase      
# 20.000% (2.0e-01)   0.0191              0.0215              Refinement Phase      
# 40.000% (4.0e-01)   0.0039              0.0000              Refinement Phase      
# 60.000% (6.0e-01)   0.0011              0.0000              Refinement Phase      
# 79.999% (8.0e-01)   0.0002              0.0000              Refinement Phase      
# 100.000% (1.0e+00)  0.0000              0.0000              Refinement Phase      
# ----------------------------------------------------------------------------------