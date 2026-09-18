import pickle
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
try:
    import scienceplots
    plt.style.use(["science", "grid"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.colors as mcolors

FONTSIZE = 16
SAVE_FOLDER = "./"

def plot_fig6(length="32"):
    data_file = f"{SAVE_FOLDER}processed_data_components_{length}_1000.pkl"
    try:
        with open(data_file, 'rb') as f:
            data_components = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return

    output_pdf = f"{SAVE_FOLDER}fig6_component_frequency_{length}.pdf"
    
    with PdfPages(output_pdf) as pdf:
        for model_name, model_data in data_components.items():
            sizes = model_data["size"]
            comp_counts = model_data["comp_counts"]
            comp_names = sorted(comp_counts.keys())
            counts_matrix = np.array([comp_counts[name] for name in comp_names])
            
            # --- PLOT 2: Cumulative Distribution (Lorenz Curve) ---
            # Increase width slightly to accommodate the colorbar comfortably
            fig, ax = plt.subplots(figsize=(6.6, 4.8))
            
            # 1. Expand to use all your available continuous sizes
            all_target_sizes = [
                4e-4, 
                1e-3, 2e-3, 4e-3,
                1e-2, 4e-2, 
                1e-1, 8e-1
            ]
            
            # 2. Setup a continuous Logarithmic Normalization for the colormap
            norm = mcolors.LogNorm(vmin=min(all_target_sizes), vmax=max(all_target_sizes))
            mapper = cm.ScalarMappable(norm=norm, cmap='plasma') 
            
            for ts in all_target_sizes:
                idx = (np.abs(sizes - ts)).argmin()
                freqs = counts_matrix[:, idx]
                freqs = freqs[freqs > 0]
                if len(freqs) == 0: continue
                
                sorted_freqs = np.sort(freqs)[::-1]
                cum_sum = np.cumsum(sorted_freqs) / np.sum(sorted_freqs)
                x_axis = np.linspace(0, 1, len(cum_sum))
                
                # Sample a color dynamically along the continuous scale
                color = mapper.to_rgba(ts)
                
                # --- HIGHLIGHT CONDITION ---
                if np.isclose(ts, 1e-3):
                    # Highlight this specific line
                    ax.plot(x_axis, cum_sum, color=color, linestyle=':', linewidth=3.5, alpha=1.0, 
                            zorder=5, label=r"$s = 10^{-3}$")
                else:
                    # Draw normal lines
                    ax.plot(x_axis, cum_sum, color=color, linewidth=1.5, alpha=0.6)
            
            # Layout formatting
            ax.set_xlabel("Proportion of Components", fontsize=FONTSIZE)
            ax.set_ylabel("Cumulative Proportion of Edges", fontsize=FONTSIZE)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axhline(0.8, color='black', linestyle='--', alpha=0.2) # 80/20 reference
            ax.axvline(0.2, color='black', linestyle='--', alpha=0.2)
            
            # Add a legend specifically for the highlighted line
            ax.legend(loc='lower right', fontsize=FONTSIZE-2, framealpha=0.9)

            # 4. Add the Colorbar
            cbar = fig.colorbar(mapper, ax=ax, pad=0.03)
            cbar.set_label("Relative Trace Size", fontsize=FONTSIZE-2)
            
            # --- ADD COLORBAR PHASE ANNOTATIONS ---
            # Draw a dashed line at the threshold (10^-2)
            cbar.ax.axhline(1e-2, color='white', linewidth=2.5, linestyle='--')
            cbar.ax.axhline(1e-2, color='black', linewidth=1.5, linestyle='--') # Gives it contrast
            
            # Place "Construction" text in the lower log-space (e.g., around 2e-3)
            # The 'plasma' colormap is dark purple/pink here, so white text reads best
            cbar.ax.text(0.5, 2e-3, "Construction", color='white', weight='bold',
                         ha='center', va='center', rotation=90, fontsize=FONTSIZE-4)
            
            # Place "Refinement" text in the upper log-space (e.g., around 1e-1)
            # The 'plasma' colormap is yellow/orange here, so black text reads best
            cbar.ax.text(0.5, 1e-1, "Refinement", color='black', weight='bold',
                         ha='center', va='center', rotation=90, fontsize=FONTSIZE-4)
            
            # 5. Place the model info as a clean top-left title
            ax.text(0.96, 0.2, model_name, transform=ax.transAxes, 
             fontsize=FONTSIZE-4, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))

            ax.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"Plots saved to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=str, default="40") # Cast default to string to match function
    args = parser.parse_args()
    plot_fig6(length=args.length)