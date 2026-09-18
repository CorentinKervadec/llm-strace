import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import gaussian_kde

# Optional: Keep scienceplots if you have it installed
try:
    import scienceplots
    plt.style.use(["science", "grid"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

FONTSIZE = 20
TICK_FONTSIZE = 12
SAVE_FOLDER = "./"

MINIMAL_CORE_SIZE = 1e-3
MINIMAL_CORE_TEXT = "Minimal Core"

def plot_fig4(length="32"):
    topo_file = f"{SAVE_FOLDER}processed_data_topology_{length}.pkl"
    agg_file = f"{SAVE_FOLDER}processed_data_agg_{length}.pkl"
    
    try:
        with open(topo_file, 'rb') as f:
            topo_data = pickle.load(f)
        with open(agg_file, 'rb') as f:
            agg_data = pickle.load(f)
            nucleus_stats = agg_data.get("nucleus", {})
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. {e}")
        return
        
    output_pdf = f"{SAVE_FOLDER}fig4_layer_composition_with_nu_{length}.pdf"
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    # Layer Labels and Colorblind Palette (Okabe-Ito)
    layer_labels = ['Initial Layers', 'Early-mid Layers', 'Late-mid Layers', 'Last Layers']
    layer_colors = ['#0072B2', '#56B4E9', '#009E73', '#E69F00'] 
    
    # Nucleus Targets and distinct colorblind colors
    nu_targets = [1, 40, 80]
    nu_colors = ['#000000', '#D55E00', '#332288'] 
    
    print(f"Generating multipage PDF for Length {length} with marginal distributions (Mode)...")
    
    with PdfPages(output_pdf) as pdf:
        for model_name, model_topo in topo_data.items():
            # fig = plt.figure(figsize=(8, 4.8))
            fig = plt.figure(figsize=(12, 4.5)) 
            
            # --- Create GridSpec for Top (Main) and Bottom (Distribution) axes ---
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1.5], hspace=0.08)
            ax_main = fig.add_subplot(gs[0])
            
            sizes = model_topo["size"]
            layer_props = model_topo["layer_props"]
            
            # 1. Plot Main Stackplot on Top Axis
            ax_main.stackplot(sizes, layer_props, labels=layer_labels, colors=layer_colors, alpha=0.8, edgecolors='none')

            # 2. Add Nucleus Distributions on Bottom Axis
            if model_name in nucleus_stats:
                model_nu = nucleus_stats[model_name]
                for i, k in enumerate(nu_targets):
                    if k in model_nu and len(model_nu[k]) > 0:
                        k_sizes = np.array(model_nu[k])
                        color = nu_colors[i % len(nu_colors)]
                        
                        # --- Calculate the Mode (Peak) in log-space ---
                        if len(k_sizes) > 1 and np.var(k_sizes) > 0:
                            log_k = np.log10(k_sizes)
                            kde = gaussian_kde(log_k)
                            # Evaluate KDE over a dense grid to find the max density
                            x_eval = np.linspace(log_k.min(), log_k.max(), 1000)
                            kde_y = kde(x_eval)
                            mode_log = x_eval[np.argmax(kde_y)]
                            mode_size = 10**mode_log  # Convert back to linear space for plotting
                        else:
                            # Fallback if there is only 1 point or zero variance
                            mode_size = k_sizes[0]
                        
                        # Draw a median Line through the axis
                        ax_main.axvline(np.median(k_sizes), color=color, linestyle='--', linewidth=2,  label=f'k={k}')

            # --- Formatting Top Axis (Main) ---
            ax_main.set_xscale('log')
            ax_main.set_xlim(1e-5, 1)
            ax_main.set_ylim(0, 1)
            ax_main.set_ylabel("Proportion of Edges", fontsize=FONTSIZE)
            ax_main.set_xlabel("Relative Trace Size", fontsize=FONTSIZE)
            ax_main.grid(True, linestyle=':', alpha=0.5)
            
            # Grab all handles and labels from the plot
            handles_main, labels_main = ax_main.get_legend_handles_labels()

            # The first 4 handles are the stackplot layers. Reverse them for top-to-bottom order.
            layer_h, layer_l = handles_main[:4][::-1], labels_main[:4][::-1]

            # Anything after the first 4 handles are your new axvline nucleus indicators
            nu_h, nu_l = handles_main[4:], labels_main[4:]

            # Combine them and create the legend
            ax_main.legend(layer_h + nu_h, layer_l + nu_l, 
                        title_fontsize=FONTSIZE-2,
                        fontsize=FONTSIZE-6,          
                        frameon=False, 
                        loc='lower center',           
                        bbox_to_anchor=(0.5, 1.02),   
                        ncol=7,                       
                        columnspacing=0.7,            
                        handletextpad=0.3,            
                        handlelength=1.2)             
            
            ax_main.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

            ax_main.text(0.95, 0.95, model_name, transform=ax_main.transAxes, 
             fontsize=FONTSIZE-4, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"Success! Saved to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', default="40", type=str)
    args = parser.parse_args()
    
    plot_fig4(length=args.length)