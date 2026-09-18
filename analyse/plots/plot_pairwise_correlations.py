import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

import scienceplots
plt.style.use(["science", "grid"])

# Your raw text data
data_text = """
Llama-3.1-8B vs Mistral-7B-v0.1
   -> Rho: +0.6859 | p-value: 0.0000e+00 | N: 5000
Llama-3.1-8B vs OLMo-2-1124-13B
   -> Rho: +0.5435 | p-value: 0.0000e+00 | N: 5000
Llama-3.1-8B vs OLMo-2-1124-7B
   -> Rho: +0.5328 | p-value: 0.0000e+00 | N: 4949
Llama-3.1-8B vs Qwen2.5-14B
   -> Rho: +0.6247 | p-value: 0.0000e+00 | N: 4309
Llama-3.1-8B vs Qwen2.5-7B
   -> Rho: +0.6438 | p-value: 0.0000e+00 | N: 5000
Llama-3.1-8B vs Qwen3-14B-Base
   -> Rho: +0.3980 | p-value: 1.7385e-189 | N: 5000
Llama-3.1-8B vs Qwen3-8B-Base
   -> Rho: +0.4284 | p-value: 2.2551e-219 | N: 4931
Llama-3.1-8B vs deepseek-llm-7b-base
   -> Rho: +0.6074 | p-value: 0.0000e+00 | N: 4900
Llama-3.1-8B vs phi-4
   -> Rho: +0.6564 | p-value: 0.0000e+00 | N: 5000
Mistral-7B-v0.1 vs OLMo-2-1124-13B
   -> Rho: +0.4393 | p-value: 4.7172e-235 | N: 5000
Mistral-7B-v0.1 vs OLMo-2-1124-7B
   -> Rho: +0.4581 | p-value: 2.0660e-255 | N: 4949
Mistral-7B-v0.1 vs Qwen2.5-14B
   -> Rho: +0.5594 | p-value: 0.0000e+00 | N: 4309
Mistral-7B-v0.1 vs Qwen2.5-7B
   -> Rho: +0.5796 | p-value: 0.0000e+00 | N: 5000
Mistral-7B-v0.1 vs Qwen3-14B-Base
   -> Rho: +0.2930 | p-value: 1.3865e-99 | N: 5000
Mistral-7B-v0.1 vs Qwen3-8B-Base
   -> Rho: +0.3593 | p-value: 3.2656e-150 | N: 4931
Mistral-7B-v0.1 vs deepseek-llm-7b-base
   -> Rho: +0.6725 | p-value: 0.0000e+00 | N: 4900
Mistral-7B-v0.1 vs phi-4
   -> Rho: +0.5831 | p-value: 0.0000e+00 | N: 5000
OLMo-2-1124-13B vs OLMo-2-1124-7B
   -> Rho: +0.6284 | p-value: 0.0000e+00 | N: 4949
OLMo-2-1124-13B vs Qwen2.5-14B
   -> Rho: +0.5767 | p-value: 0.0000e+00 | N: 4309
OLMo-2-1124-13B vs Qwen2.5-7B
   -> Rho: +0.5976 | p-value: 0.0000e+00 | N: 5000
OLMo-2-1124-13B vs Qwen3-14B-Base
   -> Rho: +0.4999 | p-value: 0.0000e+00 | N: 5000
OLMo-2-1124-13B vs Qwen3-8B-Base
   -> Rho: +0.4982 | p-value: 8.7294e-308 | N: 4931
OLMo-2-1124-13B vs deepseek-llm-7b-base
   -> Rho: +0.3771 | p-value: 2.3084e-165 | N: 4900
OLMo-2-1124-13B vs phi-4
   -> Rho: +0.5144 | p-value: 0.0000e+00 | N: 5000
OLMo-2-1124-7B vs Qwen2.5-14B
   -> Rho: +0.5501 | p-value: 0.0000e+00 | N: 4274
OLMo-2-1124-7B vs Qwen2.5-7B
   -> Rho: +0.5659 | p-value: 0.0000e+00 | N: 4949
OLMo-2-1124-7B vs Qwen3-14B-Base
   -> Rho: +0.4830 | p-value: 9.9210e-288 | N: 4949
OLMo-2-1124-7B vs Qwen3-8B-Base
   -> Rho: +0.5033 | p-value: 0.0000e+00 | N: 4881
OLMo-2-1124-7B vs deepseek-llm-7b-base
   -> Rho: +0.3904 | p-value: 2.5615e-176 | N: 4849
OLMo-2-1124-7B vs phi-4
   -> Rho: +0.5072 | p-value: 0.0000e+00 | N: 4949
Qwen2.5-14B vs Qwen2.5-7B
   -> Rho: +0.7105 | p-value: 0.0000e+00 | N: 4309
Qwen2.5-14B vs Qwen3-14B-Base
   -> Rho: +0.4718 | p-value: 8.8973e-238 | N: 4309
Qwen2.5-14B vs Qwen3-8B-Base
   -> Rho: +0.4436 | p-value: 1.9738e-204 | N: 4250
Qwen2.5-14B vs deepseek-llm-7b-base
   -> Rho: +0.4783 | p-value: 8.8304e-241 | N: 4229
Qwen2.5-14B vs phi-4
   -> Rho: +0.6317 | p-value: 0.0000e+00 | N: 4309
Qwen2.5-7B vs Qwen3-14B-Base
   -> Rho: +0.5253 | p-value: 0.0000e+00 | N: 5000
Qwen2.5-7B vs Qwen3-8B-Base
   -> Rho: +0.5369 | p-value: 0.0000e+00 | N: 4931
Qwen2.5-7B vs deepseek-llm-7b-base
   -> Rho: +0.5314 | p-value: 0.0000e+00 | N: 4900
Qwen2.5-7B vs phi-4
   -> Rho: +0.5918 | p-value: 0.0000e+00 | N: 5000
Qwen3-14B-Base vs Qwen3-8B-Base
   -> Rho: +0.5986 | p-value: 0.0000e+00 | N: 4931
Qwen3-14B-Base vs deepseek-llm-7b-base
   -> Rho: +0.2631 | p-value: 2.1562e-78 | N: 4900
Qwen3-14B-Base vs phi-4
   -> Rho: +0.3882 | p-value: 1.6636e-179 | N: 5000
Qwen3-8B-Base vs deepseek-llm-7b-base
   -> Rho: +0.3210 | p-value: 3.2953e-116 | N: 4832
Qwen3-8B-Base vs phi-4
   -> Rho: +0.4113 | p-value: 1.0902e-200 | N: 4931
deepseek-llm-7b-base vs phi-4
   -> Rho: +0.5268 | p-value: 0.0000e+00 | N: 4900
"""

# Extract relationships and parse raw text
lines = [l.strip() for l in data_text.strip().split('\n')]
models = set()
data = []

for i in range(0, len(lines), 2):
    if i+1 >= len(lines): break
    
    models_part = lines[i]
    stats_part = lines[i+1]
    
    model_a, model_b = models_part.split(' vs ')
    models.add(model_a)
    models.add(model_b)
    
    # Extract Rho
    rho_match = re.search(r'Rho: \+?(-?\d+\.\d+)', stats_part)
    rho = float(rho_match.group(1))
    
    data.append((model_a, model_b, rho))

# Build empty dataframe
models = sorted(list(models))
df = pd.DataFrame(index=models, columns=models, dtype=float)

# Fill diagonal with 1.0 (perfect self-correlation)
for m in models:
    df.loc[m, m] = 1.0

# Fill matrix symmetrically
for m1, m2, rho in data:
    df.loc[m1, m2] = rho
    df.loc[m2, m1] = rho

# ----- Plotting -----
plt.figure(figsize=(7, 6))

# Clean up model names to make the axes less cluttered 
display_names = [m.replace('-Base', '').replace('-base', '') for m in models]
df.index = display_names
df.columns = display_names

# Use a colormap that is accessible and common in NLP/ML papers
cmap = sns.color_palette("Blues", as_cmap=True)

# Generate a mask for the upper triangle to make the plot cleaner
mask = np.triu(np.ones_like(df, dtype=bool), k=1)

ax = sns.heatmap(df, mask=mask, annot=True, fmt=".2f", cmap=cmap, 
            vmin=0.2, vmax=1.0, square=True, linewidths=.5, 
            cbar_kws={"shrink": .8, "label": "Spearman Correlation"},
            annot_kws={"size": 8})

# Fix the formatting so text isn't cut off
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
# plt.title("Correlation (Rho) across Models", fontsize=12, pad=15)

plt.tight_layout()
plt.savefig("heatmap_rho.pdf", bbox_inches='tight')
plt.show()