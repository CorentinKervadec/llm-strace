
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import argparse
import pandas as pd

def analyze_data(file_path, output_path):
    # Load the TSV file
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    sns.set_style("whitegrid")

    with PdfPages(output_path) as pdf:
        print(f"Generating plots to {output_path}...")

        # --- 1. Regression Plots ---
        plot_pairs = [
            ('entropy', 'auc_tv'),
            ('step', 'auc_tv'),
            ('next_token_prob', 'auc_tv')
        ]

        for x_col, y_col in plot_pairs:
            if x_col not in df.columns or y_col not in df.columns:
                print(f"Skipping {x_col} vs {y_col}: Column not found.")
                continue
            
            # Remove NaNs for stats calculation
            clean_data = df[[x_col, y_col]].dropna()
            if len(clean_data) < 2:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation
            r, p = stats.pearsonr(clean_data[x_col], clean_data[y_col])

            # Plot regression
            sns.regplot(data=df, x=x_col, y=y_col, ax=ax, 
                        scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

            # Annotation
            stats_text = f'r = {r:.3f}\np = {p:.3e}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'Regression: {y_col} vs {x_col}', fontsize=16)
            ax.set_xlabel(x_col, fontsize=14)
            ax.set_ylabel(y_col, fontsize=14)
            
            pdf.savefig(fig)
            plt.close(fig)

        # --- 2. Distribution of Entropy and Gen Entropy ---
        entropy_cols = ['entropy', 'gen_entropy']
        existing_cols = [c for c in entropy_cols if c in df.columns]
        
        if existing_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Melt the dataframe to plot both distributions on the same axis easily
            df_melt = df[existing_cols].melt(var_name='Metric', value_name='Value')
            
            sns.histplot(data=df_melt, x='Value', hue='Metric', 
                         kde=True, element="step", ax=ax, alpha=0.5)
            
            ax.set_title('Distribution of Entropy vs Gen Entropy', fontsize=16)
            ax.set_xlabel('Entropy', fontsize=14)
            
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("Skipping Entropy distribution: columns not found.")

        # --- 3. Distribution of AUC ---
        if 'auc_tv' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.histplot(data=df, x='auc_tv', kde=True, ax=ax, color='purple', alpha=0.6)
            
            ax.set_title('Distribution of AUC (auc_tv)', fontsize=16)
            ax.set_xlabel('AUC', fontsize=14)
            
            pdf.savefig(fig)
            plt.close(fig)
        else:
            print("Skipping AUC distribution: auc_tv column not found.")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Generation & Extraction - Plotting")
    parser.add_argument('--data', type=str, required=True, help="Path to input TSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to output PDF file")
    
    args = parser.parse_args()
    
    analyze_data(args.data, args.output)