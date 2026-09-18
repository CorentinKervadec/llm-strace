import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Keep your TeX path if needed
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
try:
    import scienceplots
    plt.style.use(["science", "grid"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

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

# --- HIGHLIGHT CONFIGURATION ---
# Adjusted to apply to the Y-axis (Relative Graph Size)
TRANSITION_START = 1e-4
TRANSITION_END = 1e-2
TRANSITION_ANNOTATION_TEXT = "Construction Phase"

REFINEMENT_START = 1e-2
REFINEMENT_END = 1
REFINEMENT_ANNOTATION_TEXT = "Refinement Phase"

def plot_fig2_double_column_errorbars(data_file="processed_data_agg_40.pkl"):
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)["nucleus"]
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return
        
    # Wide figure optimized for a double-column span
    fig, ax = plt.subplots(figsize=(12, 4.5)) 
    
    # 1. Add the Shaded Highlight Region horizontally
    ax.axhspan(TRANSITION_START, TRANSITION_END, color='#F0E442', alpha=0.2, lw=0)
    ax.axhspan(REFINEMENT_START, REFINEMENT_END, color="#42CAF0", alpha=0.2, lw=0)
    
    k_values = [1, 5, 10, 20, 40, 60, 80, 90]
    
    for i, (model_name, model_data) in enumerate(data.items()):
        color = COLORS[i % len(COLORS)]
        
        medians = []
        lower_err = []
        upper_err = []
        
        # Calculate Medians and Error Bounds (25th and 75th percentiles)
        for k in k_values:
            k_data = np.array(model_data[k])
            if len(k_data) > 0:
                med = np.median(k_data)
                medians.append(med)
                
                # Matplotlib error bars require relative offsets from the median
                # e.g., median - 25th percentile, and 75th percentile - median
                lower_err.append(med - np.percentile(k_data, 25))
                upper_err.append(np.percentile(k_data, 75) - med)
            else:
                medians.append(np.nan)
                lower_err.append(np.nan)
                upper_err.append(np.nan)
                
        # Plot using standard error bars and dashed lines to prevent implying continuous data
        ax.errorbar(k_values, medians, yerr=[lower_err, upper_err], 
                    label=model_name, color=color, 
                    linewidth=2, linestyle='-', marker='o', markersize=6, 
                    capsize=4, capthick=2, alpha=0.9)
        
    # Axis formatting
    ax.set_yscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels([f"{k}\%" for k in k_values]) # Kept TeX escaping
    
    ax.set_xlabel("Nucleus Target ($k$)", fontsize=FONTSIZE)
    ax.set_ylabel("Required Relative Graph Size", fontsize=FONTSIZE)
    
    # Grid formatting
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3)
    
    # 2. Add the Text Annotations pointing to the regions
    # Centered at log-midpoints of the y-axis regions (1e-3 and 1e-1)
    ax.annotate(TRANSITION_ANNOTATION_TEXT, 
                xy=(70, 1e-2),              # Arrow tip coordinates (x, y)
                xytext=(70, 3e-4),          # Text box coordinates (x, y)
                fontsize=FONTSIZE-2,
                fontweight='bold',
                color='#333333',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8))
                
    ax.annotate(REFINEMENT_ANNOTATION_TEXT, 
                xy=(60, 1e-1),              # Arrow tip coordinates (x, y)
                xytext=(5, 2e-1),          # Text box coordinates (x, y)
                fontsize=FONTSIZE-2,
                fontweight='bold',
                color='#333333',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8))
    
    # Clean Legend outside the plot area
    ax.legend(
        fontsize=FONTSIZE-3,          # You can drop this to FONTSIZE-4 if strictly needed
        frameon=False, 
        loc='lower center',           
        bbox_to_anchor=(0.5, 1.02),   
        ncol=5,                       # 3 columns forces 4 rows (much narrower)
        columnspacing=0.8,            # (Trick 1) Squeezes the text columns closer together
        handletextpad=0.4,            # (Trick 2) Reduces the gap between the line sample and the text
        handlelength=1.2              # (Trick 3) Makes the colored line sample itself shorter
    )
    
    plt.tight_layout()
    output_filename = "fig2_nucleus_size.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_fig2_double_column_errorbars()