import pickle
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import scienceplots
plt.style.use(["science", "grid"])

FONTSIZE = 16

# A 10-color accessible palette based on Seaborn's 'colorblind' standard
COLORS = [
    '#0173b2', # Blue
    '#de8f05', # Orange
    '#029e73', # Green
    '#d55e00', # Red/Rust
    '#cc78bc', # Purple
    '#ca9161', # Brown
    '#fbafe4', # Pink
    '#949494', # Gray
    '#ece133', # Yellow
    '#56b4e9'  # Light Blue
]

def plot_figA(data_file="processed_data_agg_40.pkl"):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)["aggregate"]
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for idx, (model_name, metrics) in enumerate(data.items()):
        color = COLORS[idx % len(COLORS)]
        # Trace line
        p = ax.plot(metrics["size"], metrics["tv_trace"], label=f"{model_name} (Trace)", linewidth=2, color=color)
        # Random line with identical color but dashed
        ax.plot(metrics["size"], metrics["tv_random"], label=f"{model_name} (Random)", 
                linewidth=2, linestyle='--', color=color)
        
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative Trace Size", fontsize=FONTSIZE)
    ax.set_ylabel("Reconstruction Error (TV)", fontsize=FONTSIZE)
    # ax.set_title("Trace Extraction vs. Baseline (TV Comparison)", fontsize=FONTSIZE+2)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(
        fontsize=FONTSIZE-3,          # You can drop this to FONTSIZE-4 if strictly needed
        frameon=False, 
        loc='lower center',           
        bbox_to_anchor=(0.5, 1.02),   
        ncol=3,                       # 3 columns forces 4 rows (much narrower)
        columnspacing=0.8,            # (Trick 1) Squeezes the text columns closer together
        handletextpad=0.4,            # (Trick 2) Reduces the gap between the line sample and the text
        handlelength=1.2              # (Trick 3) Makes the colored line sample itself shorter
    )
    
    plt.tight_layout()
    plt.savefig("figA_trace_vs_random.pdf")
    plt.show()

if __name__ == "__main__":
    plot_figA()