import pickle
import matplotlib.pyplot as plt
import os
import math

# Keep TeX path if needed
if '/Library/TeX/texbin' not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

try:
    import scienceplots
    plt.style.use(["science", "grid"])
except Exception:
    print("scienceplots not found. Proceeding with default style.")

FONTSIZE = 14 
TICK_FONTSIZE = 12

# Colorblind-friendly palette mapped specifically to languages
LANGS = ['en', 'ar', 'es', 'ru']
LANG_LABELS = {
    'en': 'English (en)',
    'ar': 'Arabic (ar)',
    'es': 'Spanish (es)',
    'ru': 'Russian (ru)'
}
LANG_COLORS = {
    'en': '#d55e00',  # Red / Vermillion
    'ar': '#0173b2',  # Blue
    'es': '#de8f05',  # Orange
    'ru': '#029e73'   # Green
}

TRANSITION_START = 1e-4
TRANSITION_END = 1e-2
REFINEMENT_START = 1e-2
REFINEMENT_END = 1

def interpolate_logx(x_vec, y_vec, target_x):
    """Linearly interpolates y-values in log10 space of x."""
    if x_vec is None or y_vec is None or len(x_vec) == 0 or len(y_vec) == 0: 
        return float('nan')
    if target_x <= x_vec[0]: 
        return y_vec[0]
    if target_x >= x_vec[-1]: 
        return y_vec[-1]
    
    log_x = [math.log10(x) if x > 0 else -20 for x in x_vec]
    log_target = math.log10(target_x) if target_x > 0 else -20
    
    for i in range(len(log_x) - 1):
        if log_x[i] <= log_target <= log_x[i+1]:
            x0, x1 = log_x[i], log_x[i+1]
            y0, y1 = y_vec[i], y_vec[i+1]
            if x1 == x0: 
                return y0
            return y0 + (y1 - y0) * (log_target - x0) / (x1 - x0)
    return float('nan')

def plot_multiling_grid(
    data_file="processed_data_agg_multiling_40_500.pkl", 
    english_data_file="processed_data_agg_40.pkl"
):
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)["aggregate"]
    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
        return

    # Load English data
    english_data = {}
    if os.path.exists(english_data_file):
        try:
            with open(english_data_file, 'rb') as f:
                english_data = pickle.load(f)["aggregate"]
        except Exception as e:
            print(f"Warning: Could not load English data from {english_data_file}: {e}")
    else:
        print(f"Warning: {english_data_file} not found. English curves will be omitted.")
    
    models = list(data.keys())
    n_models = len(models)
    
    if n_models == 0:
        print("No models found in the data.")
        return

    # Calculate Grid Layout (Max 3 columns)
    n_cols = min(3, n_models)
    n_rows = math.ceil(n_models / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Store results for the ASCII table summary
    table_rows = []
    landmarks = [1e-4, 1e-2, 1.0]
    
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        
        # 1. Background highlights
        ax.axvspan(TRANSITION_START, TRANSITION_END, color='#F0E442', alpha=0.3, lw=0)
        ax.axvspan(REFINEMENT_START, REFINEMENT_END, color="#42CAF0", alpha=0.3, lw=0)
        
        # Combine English data with multilingual data for this model
        model_data = {}
        if model_name in english_data:
            model_data['en'] = english_data[model_name]
        model_data.update(data[model_name])
        
        # Order languages: English first, then defined multilingual order, then any remaining
        ordered_langs = [l for l in LANGS if l in model_data] + [l for l in model_data if l not in LANGS]

        # 2. Plot each language
        for lang in ordered_langs:
            metrics = model_data[lang]
            if "size" in metrics and "tv_trace" in metrics:
                display_label = LANG_LABELS.get(lang, lang)
                ax.plot(metrics["size"], metrics["tv_trace"], 
                        label=display_label, 
                        linewidth=2.5,    
                        color=LANG_COLORS.get(lang, '#000000'), 
                        alpha=0.9,        
                        solid_capstyle='round') 
                
                # Extract numerical values at transition boundaries for summary table
                vals = [interpolate_logx(metrics["size"], metrics["tv_trace"], q) for q in landmarks]
                table_rows.append((model_name, lang, vals))
        
        # 3. Aesthetics
        ax.set_title(model_name, fontsize=FONTSIZE, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(-0.05, 1.05) 
        
        # Labels on bottom row and leftmost column only
        if idx >= n_models - n_cols:
            ax.set_xlabel("Relative Trace Size", fontsize=FONTSIZE)
        if idx % n_cols == 0:
            ax.set_ylabel("Reconstruction Error (TV)", fontsize=FONTSIZE)
        
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Hide unused subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    # 4. Generate shared legend at the top (deduplicated across axes)
    by_label = {}
    for i in range(n_models):
        h, l = axes[i].get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in by_label:
                by_label[label] = handle

    fig.legend(by_label.values(), by_label.keys(), 
               fontsize=FONTSIZE-2,          
               frameon=False, 
               loc='lower center',           
               bbox_to_anchor=(0.5, 1.02),   
               ncol=len(by_label),
               columnspacing=1.5,
               handletextpad=0.4)
    
    plt.tight_layout()
    output_filename = "fig_trace_vs_tv_multiling_grid.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved multilingual grid plot to {output_filename}\n")
    
    # 5. Generate and print ASCII summary table
    print("=" * 58)
    print(f"{'Model':<20} {'Lang':<8} {'TV@1e-4':<9} {'TV@1e-2':<9} {'TV@1.0':<9}")
    print("=" * 58)
    
    prev_model = ""
    for model, lang, vals in table_rows:
        model_col = model if model != prev_model else ""
        v_str = [f"{v:.3f}" if not math.isnan(v) else "N/A" for v in vals]
        print(f"{model_col:<20} {lang:<8} {v_str[0]:<9} {v_str[1]:<9} {v_str[2]:<9}")
        prev_model = model
        
    print("=" * 58)
    plt.show()

if __name__ == "__main__":
    plot_multiling_grid(
        data_file="./multiling_data/processed_data_agg_multiling_40_5000.pkl",
        english_data_file="processed_data_agg_40.pkl"
    )