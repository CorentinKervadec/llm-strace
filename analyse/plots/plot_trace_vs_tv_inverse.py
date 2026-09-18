import pickle
import matplotlib.pyplot as plt
import os

# Keep your TeX path if needed
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import scienceplots
plt.style.use(["science", "grid"])


# Optimized font sizes for a smaller aspect ratio
FONTSIZE = 14 
TICK_FONTSIZE = 12

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
# Adjust these x-values to exactly match where the sharp drop happens in your data
TRANSITION_START = 1e-4
# TRANSITION_END = 1e-2
# TRANSITION_ANNOTATION_TEXT = "Transition Phase"

# REFINEMENT_START = 1e-2
# REFINEMENT_END = 1
# REFINEMENT_ANNOTATION_TEXT = "Refinement Phase"

def plot_fig1_teaser(data_file="processed_data_agg_40_5000.pkl"):
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)["aggregate"]
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return
        
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    
    # 1. Add the Shaded Highlight Region (Do this before plotting lines so it stays in the background)
    # ax.axvspan(TRANSITION_START, TRANSITION_END, color='#F0E442', alpha=0.3, lw=0)
    # ax.axvspan(REFINEMENT_START, REFINEMENT_END, color="#42CAF0", alpha=0.3, lw=0)
    
    # Plot each model
    for idx, (model_name, metrics) in enumerate(data.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(metrics["size"], metrics["tv_inv"], 
                label=model_name, 
                linewidth=3.5,    
                color=color, 
                alpha=0.9,        
                solid_capstyle='round') 
        
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(-0.05, 1.05) 
    
    ax.set_xlabel("Relative Trace Size", fontsize=FONTSIZE)
    ax.set_ylabel("Reconstruction Error (TV)", fontsize=FONTSIZE)
    
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle=':', alpha=0.6)
    
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
    
    # plt.tight_layout()
    output_filename = "fig_trace_vs_tv_inverse.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved teaser to {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_fig1_teaser()