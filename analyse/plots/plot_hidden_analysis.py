import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages

# --- Visual Style Integration ---
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
import scienceplots
plt.style.use(["science", "grid"])

# Optimized font sizes
FONTSIZE = 14 
TICK_FONTSIZE = 12

COLORS = [
    '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', 
    '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'
]
CUSTOM_CMAP = ListedColormap(COLORS)

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate publication-ready figures.")
    parser.add_argument('--data_dir', type=str, default="processed_data", help="Directory where processed metrics are saved")
    parser.add_argument('--output_plots_dir', type=str, default="analysis_plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_plots_dir, exist_ok=True)

    matrix_file = os.path.join(args.data_dir, 'aggregated_matrices.npz')
    tsne_file = os.path.join(args.data_dir, 'tsne_embeddings_summary.tsv')
    
    if not (os.path.exists(matrix_file) and os.path.exists(tsne_file)):
        raise FileNotFoundError("Missing target metrics files inside data directory. Run Step 1 first!")

    matrices = np.load(matrix_file)
    df_tsne = pd.read_csv(tsne_file, sep='\t')
    
    mean_rel_sizes = matrices['mean_rel_sizes']
    axis_labels = [f"{val:.1e}" for val in mean_rel_sizes]

    # Keep every 2nd label visible; set others to empty strings
    sparse_axis_labels = [label if i % 2 == 0 else "" for i, label in enumerate(axis_labels)]

    configs = ['trace_only', 'random_only', 'trace_inverse']

    # -----------------------------------------------------------------
    # PLOT SET 1: Geometry & Information Imbalance Matrices
    # -----------------------------------------------------------------
    print("Generating Geometry Analysis Matrices PDF (incl. Info Imbalance)...")
    geometry_pdf_path = os.path.join(args.output_plots_dir, 'geometry_analysis_all_configs.pdf')
    with PdfPages(geometry_pdf_path) as pdf:
        for cfg in configs:
            fig, axes = plt.subplots(1, 4, figsize=(28, 6))
            
            # 1. Cosine Distance
            sns.heatmap(matrices[f'{cfg}_mean_cosine_dist'], ax=axes[0], xticklabels=axis_labels, yticklabels=axis_labels, cmap="magma")
            axes[0].set_title(r"Cosine Distance", fontsize=FONTSIZE, weight='bold')

            # 2. L2 Distance
            sns.heatmap(matrices[f'{cfg}_mean_l2_dist'], ax=axes[1], xticklabels=axis_labels, yticklabels=axis_labels, cmap="mako")
            axes[1].set_title(r"L2 Distance", fontsize=FONTSIZE, weight='bold')

            # 3. CKA Alignment
            sns.heatmap(matrices[f'{cfg}_cka_matrix'], ax=axes[2], xticklabels=axis_labels, yticklabels=axis_labels, cmap="viridis", vmin=0, vmax=1)
            axes[2].set_title(r"CKA Alignment", fontsize=FONTSIZE, weight='bold')
            
            # 4. Information Imbalance
            sns.heatmap(matrices[f'{cfg}_ii_matrix'], ax=axes[3], xticklabels=axis_labels, yticklabels=axis_labels, cmap="inferno", norm=LogNorm())
            axes[3].set_title(r"Info. Imbalance", fontsize=FONTSIZE, weight='bold')

            for ax in axes:
                ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
                ax.set_xlabel("Trace Size", fontsize=FONTSIZE-2)

            plt.suptitle(f"Geometry and Structural Analysis - {cfg.replace('_', ' ').title()}", fontsize=FONTSIZE+4, weight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    # -----------------------------------------------------------------
    # PLOT SET 1.5: CKA Analysis Only (Trace & Random)
    # -----------------------------------------------------------------
    print("Generating CKA-only PDF for Trace and Random configs...")
    cka_pdf_path = os.path.join(args.output_plots_dir, 'cka_analysis_only.pdf')
    cka_configs = ['trace_only', 'random_only']
    config_labels = {'trace_only': 'L1', 'random_only': 'Random'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, cfg in enumerate(cka_configs):
        sns.heatmap(matrices[f'{cfg}_cka_matrix'], ax=axes[idx], xticklabels=sparse_axis_labels, yticklabels=sparse_axis_labels, cmap="viridis", vmin=0, vmax=1)
        axes[idx].set_title(f"CKA Alignment - {config_labels[cfg]}", fontsize=FONTSIZE, weight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        axes[idx].set_xlabel("Trace Size", fontsize=FONTSIZE-2)

    plt.tight_layout()
    fig.savefig(cka_pdf_path, bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------------------
    # PLOT SET 2: Trajectory Dynamics (Velocity, Cosine, Curvature, Magnitude)
    # -----------------------------------------------------------------
    print("Generating Trajectory Dynamics Profiles (4-Panel Grid)...")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 5.5))
    
    config_labels = {'trace_only': 'Trace Only', 'random_only': 'Random Only', 'trace_inverse': 'Trace Inverse'}
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvspan(1e-4, 1e-2, color='#F0E442', alpha=0.2, lw=0)
        ax.axvspan(1e-2, 1.0, color="#42CAF0", alpha=0.2, lw=0)
        ax.set_xscale('log')
        ax.set_xlim(1e-5, 1)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("Relative Trace Size", fontsize=FONTSIZE)

    x_intervals = 0.5 * (mean_rel_sizes[:-1] + mean_rel_sizes[1:]) 
    x_curvature = mean_rel_sizes[1:-1]
    
    for idx, cfg in enumerate(configs):
        color = COLORS[idx % len(COLORS)]
        
        # Velocity
        v_data = matrices[f'{cfg}_trajectory_velocity']
        ax1.plot(x_intervals, v_data, label=config_labels[cfg], linewidth=3.5, color=color, alpha=0.8)
            
        # Cosine Distance
        cos_data = matrices[f'{cfg}_trajectory_cosine']
        ax2.plot(x_intervals, cos_data, linewidth=3.5, color=color, alpha=0.8)
            
        # Curvature
        c_data = matrices[f'{cfg}_trajectory_curvature']
        ax3.plot(x_curvature, c_data, linewidth=3.5, color=color, alpha=0.8)
        
        # Magnitude
        m_data = matrices[f'{cfg}_mean_magnitude']
        ax4.plot(mean_rel_sizes, m_data, linewidth=3.5, color=color, alpha=0.8)

    ax1.set_ylabel(r"$\Vert \Delta h \Vert$", fontsize=FONTSIZE)
    ax1.set_title("L2 Velocity", fontsize=FONTSIZE, fontweight='bold')
    
    ax2.set_ylabel(r"$1 - S_c$", fontsize=FONTSIZE)
    ax2.set_title("Cosine Dist.", fontsize=FONTSIZE, fontweight='bold')
    
    ax3.set_ylabel(r"$\kappa_M$", fontsize=FONTSIZE)
    ax3.set_title("Menger Curv.", fontsize=FONTSIZE, fontweight='bold')
    
    ax4.set_ylabel(r"$\Vert h \Vert$", fontsize=FONTSIZE)
    ax4.set_title("Magnitude", fontsize=FONTSIZE, fontweight='bold')
    
    # Legend in center
    ax2.legend(fontsize=FONTSIZE-2, frameon=False, loc='lower center', bbox_to_anchor=(1.1, 1.1), ncol=3)
    
    plt.savefig(os.path.join(args.output_plots_dir, 'trajectory_dynamics_profiles.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # -----------------------------------------------------------------
    # MARKDOWN TABLE: CKA Analysis (vs Final Trace Size)
    # -----------------------------------------------------------------
    print("Generating CKA Markdown Table...")
    
    # Isolate the final column (index -1) which represents comparison with the final trace size
    cka_trace = matrices['trace_only_cka_matrix'][:, -1]
    cka_random = matrices['random_only_cka_matrix'][:, -1]

    md_table = "| Relative Trace Size | Trace Only (vs Final) | Random Only (vs Final) |\n"
    md_table += "|---|---|---|\n"

    for i, size in enumerate(mean_rel_sizes):
        md_table += f"| {size:.1e} | {cka_trace[i]:.4f} | {cka_random[i]:.4f} |\n"

    # Save to a markdown file
    md_path = os.path.join(args.output_plots_dir, 'cka_analysis_vs_final.md')
    with open(md_path, 'w') as f:
        f.write(md_table)

    # Print the table to the console so you can copy it immediately
    print("\n--- CKA Analysis Table ---")
    print(md_table)
    print("--------------------------\n")

    print(f"Publication-ready graphs and markdown tables successfully rendered inside '{args.output_plots_dir}'.")

if __name__ == "__main__":
    main()