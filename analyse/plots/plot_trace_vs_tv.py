import pickle
import matplotlib.pyplot as plt
import os
import csv # Added for data export

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
TRANSITION_START = 1e-4
TRANSITION_END = 1e-2
TRANSITION_ANNOTATION_TEXT = "Construction Phase"

REFINEMENT_START = 1e-2
REFINEMENT_END = 1
REFINEMENT_ANNOTATION_TEXT = "Refinement Phase"

def plot_fig1_teaser(data_file="processed_data_agg_40.pkl"):
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)["aggregate"]
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return
        
    # --- RAW DATA EXPORT ---
    csv_filename = "raw_data_trace_vs_tv.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write headers
        writer.writerow(["Model", "Relative_Trace_Size", "Reconstruction_Error_TV"])
        
        # Iterate through data and write rows
        for model_name, metrics in data.items():
            sizes = metrics["size"]
            tvs = metrics["tv_trace"]
            # Zip the x and y values together to write row by row
            for s, t in zip(sizes, tvs):
                writer.writerow([model_name, s, t])
    print(f"Exported raw plot data to {csv_filename}")
    # -----------------------

    # fig, ax = plt.subplots(figsize=(12, 4.5)) 
    fig, ax = plt.subplots(figsize=(8, 6)) 
    
    # 1. Add the Shaded Highlight Region
    ax.axvspan(TRANSITION_START, TRANSITION_END, color='#F0E442', alpha=0.3, lw=0)
    ax.axvspan(REFINEMENT_START, REFINEMENT_END, color="#42CAF0", alpha=0.3, lw=0)
    
    # Plot each model
    for idx, (model_name, metrics) in enumerate(data.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(metrics["size"], metrics["tv_trace"], 
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
    
    # 2. Add the Text Annotations
    ax.annotate(TRANSITION_ANNOTATION_TEXT, 
                xy=(1e-3, 0.45),
                xytext=(1e-4, 0.25),
                fontsize=FONTSIZE-2,
                fontweight='bold',
                color='#333333',
                arrowprops=dict(
                    facecolor='#333333', 
                    edgecolor='none',
                    shrink=0.05, 
                    width=2.5, 
                    headwidth=8,
                    headlength=8,
                    connectionstyle="arc3,rad=0.2"
                ),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8))
                
    ax.annotate(REFINEMENT_ANNOTATION_TEXT, 
                xy=(5e-2, 0.2), 
                xytext=(5e-2, 0.4),
                fontsize=FONTSIZE-2,
                fontweight='bold',
                color='#333333',
                arrowprops=dict(
                    facecolor='#333333', 
                    edgecolor='none',
                    shrink=0.05, 
                    width=2.5, 
                    headwidth=8,
                    headlength=8,
                    connectionstyle="arc3,rad=0.2"
                ),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8))
    
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(
        fontsize=FONTSIZE-3,          
        frameon=False, 
        loc='lower center',           
        bbox_to_anchor=(0.5, 1.02),   
        ncol=5,                       
        columnspacing=0.8,            
        handletextpad=0.4,            
        handlelength=1.2              
    )
    
    output_filename = "fig1_trace_vs_tv.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved teaser to {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_fig1_teaser()