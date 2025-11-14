from src.llm_hooked.hook_constructors import get_hooked_constructor
from src.llm_trace.llm_trace import LLM_STRACE
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormaps
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def main():
    # input sentences
    sentences_sets = [
        [
            ("Any child learns", "responsibility"),
            ("Every child who owns pets learns", "responsibility"),
            ("A child owns a pet. The child learns", "responsibility"),
            ("Every child who owns pets that need care learns", "responsibility"),
            ("A child owns a pet. Pets need care. The child learns", "responsibility"),
            ("Among children who own pets that need care, every child learns", "responsibility"),
        ],
        [
            ("Any student", "succeeds"),
            ("Any student who reads books", "succeeds"),
            ("A student reads books. The student", "succeeds"),
            ("Any student who reads books that inspire people", "succeeds"),
            ("A student reads books. These books inspire people. The student", "succeeds"),
            ("Among students who read books that inspire people, any student", "succeeds"),
        ],
        [
            ("The family gathered by the hospital bed. He had been ill for months. The old man finally kicked the", "bucket"),
            ("He missed the goal and the crowd laughed. In a fit of rage, the angry child purposefully kicked the", "bucket")
        ],
        [
            ("The lawyer warned him to stay silent, but under pressure from the prosecution, the witness spilled the", "beans"),
            ("The paper bag had a small tear. While walking home from the grocery store, the clumsy cook spilled the", "beans")
        ],
        [
            ("The party was quiet and awkward. Wanting everyone to feel comfortable, the polite host broke the", "ice"),
            ("It was the first arctic journey of the season. With impressive power, the heavy ship broke the", "ice")
        ]
    ]
    # parameters
    model_name = 'mistralai/Mistral-7B-v0.1' # 'allenai/OLMo-2-0425-1B' #'mistralai/Mistral-7B-v0.1'
    half_precision = True
    untrained = False
    importance_mode = 'norm'
    strace_mode = 'nucleus' # 'nucleus' or 'threshold'
    batch_size = 32
    threshold_values = [1.0, .9995, .999, .995, .99, .985, .98, .975, .97, .96, .95, .9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    temp = []
    # # double the treshold_values
    # for i in range(len(threshold_values)):
    #     temp.append(threshold_values[i])
    #     if i < len(threshold_values) -1:
    #         temp.append((threshold_values[i]+threshold_values[i+1])/2)
    # threshold_values = temp
    # initialise llm_hooked
    HOOKED_CONSTRUCTOR = get_hooked_constructor(model_name)
    llm_hooked = HOOKED_CONSTRUCTOR(model_name, half_precision, untrained)
    
    all_data = []

    for sentences in sentences_sets:

        data = []

        for i, (input_sentence, gt_next) in enumerate(sentences):
            if not llm_hooked.extraction_hook_registred():
                llm_hooked.register_extraction_hooks()

            print(f"[MAIN] SENTENCE: '{input_sentence}'")
            
            """
            This part should be done in GPU
            """

            # initialise the strace
            strace = LLM_STRACE((input_sentence, gt_next), llm_hooked, track_time=False)

            # initialise the llm graph
            strace.initialize_graph(importance_mode)
            
            strace.populate_graph(batch_size)
            
            """
            This part should be done in CPU
            """
            
            strace.extract_strace(threshold_values, mode=strace_mode)
            
            # don't forget to remove the extraction hooks before doing the masking
            llm_hooked.remove_extraction_hooks()

            strace.compute_stratum_reconstruction_error()

            data.append({
                'input': input_sentence,
                'next': gt_next,
                'size': strace.strata_rel_size,
                'nucleus': strace.strata_reco_nu['trace']['only'],
                'tv_dist': strace.strata_reco_tv['trace']['only'],
                'loss': strace.strata_loss['trace']['only'],
                'entropy': strace.strata_entropy['trace']['only'],
                'nucleus_60':strace.strata_nucleus_60['trace']['only'],
            })
        all_data.append(data)
    return all_data
            
def plot_from_dict_list(all_data, output_pdf="per_sentence_plots.pdf"):
    """
    Generates per-sentence plots from a list of dictionaries, with
    filled areas, AUC calculations, and larger legend font.

    Args:
        all_data (list): A list where each item is a dictionary in
                         the format you described.
        output_pdf (str): The filename for the output PDF.
    """
    print(f"Plotting per-sentence results for {len(all_data)} sentences...")
    
    # Define the keys from your dict to plot
    metrics_to_plot = [
        ('nucleus', 'Nucleus Reconstruction'),
        ('tv_dist', 'TV Reconstruction'),
        ('loss', 'Trace Loss'),
        ('entropy', 'Trace Entropy'),
    ]
    
    num_sentences = len(all_data)
    if num_sentences == 0:
        print("No data to plot.")
        return
        
    # Get a colormap for the different sentences
    colors = cm.jet(np.linspace(0, 1, num_sentences))
    
    with PdfPages(output_pdf) as pdf:
        
        for y_key, y_label in metrics_to_plot:
            
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # --- NEW: Store handles and labels for two legends ---
            line_handles = []
            legend1_labels = [] # For AUC + Sentence
            legend2_labels = [] # For Nucleus
            
            # Plot each sentence
            for i, sentence_data in enumerate(all_data):
                
                try:
                    # Extract raw data using the keys from your dictionary
                    x_sent_raw = sentence_data['size'] 
                    y_sent_raw = sentence_data[y_key]
                    in_text = sentence_data['input']
                    next_word = sentence_data['next']
                    color = colors[i]

                    # --- New: Sort data by x-axis for correct plotting and AUC ---
                    # This is critical if 'size' is not already sorted
                    sort_indices = np.argsort(x_sent_raw)
                    x_sent_sorted = np.array(x_sent_raw)[sort_indices]
                    y_sent_sorted = np.array(y_sent_raw)[sort_indices]

                    # --- New: Calculate Area Under Curve (AUC) ---
                    # Use the sorted data for the trapezoidal rule
                    auc = np.trapezoid(y_sent_sorted, x_sent_sorted)
                    
                    # Create a truncated label for the legend
                    sent_short = (in_text[:40] + '...') if len(in_text) > 40 else in_text
                    
                    # --- New: Add AUC to the label ---
                    # This is for the FIRST legend
                    label1 = f"(AUC: {auc:.2f}) '{sent_short}' -> '{next_word}'"
                    
                    # --- NEW: Create label for the SECOND legend ---
                    nucleus_list = sentence_data.get('nucleus_60', ['N/A'])
                    last_nucleus = nucleus_list[-1] # nucleus of the last stratum (i.e. full model prediction)
                    label2 = f"Nucleus: {', '.join(last_nucleus)}"
                    
                    # --- Plot the line ---
                    # We must store the returned Line2D object (the "handle")
                    line = ax.plot(x_sent_sorted, y_sent_sorted, 
                            color=color, marker='o', 
                            markersize=3, linestyle='-')[0]
                    
                    # --- New: Add the fill area ---
                    ax.fill_between(x_sent_sorted, y_sent_sorted, 0, 
                                    color=color, alpha=0.15)
                    
                    # --- NEW: Append handles and labels to our lists ---
                    line_handles.append(line)
                    legend1_labels.append(label1)
                    legend2_labels.append(label2)
                
                except KeyError:
                    print(f"Warning: Could not find key '{y_key}' or 'size' for sentence {i}")
                except Exception as e:
                    print(f"Warning: Failed to plot sentence {i}: {e}")

            ax.set_title(f'{y_label} vs. Relative Stratum Size (Per-Sentence)')
            ax.set_xlabel('Relative Stratum Size')
            ax.set_ylabel(y_label)
            
            # --- NEW: Create and add both legends ---
            
            # 1. Create the first legend (AUC + Sentence)
            leg1 = ax.legend(line_handles, legend1_labels,
                             title="Sentence (AUC & Input)",
                             bbox_to_anchor=(1.05, 1), 
                             loc='upper left', 
                             fontsize='medium')
            # "Bake" the first legend onto the plot
            ax.add_artist(leg1)

            # 2. Create the second legend (Nucleus)
            # We re-use the line_handles but pass the new nucleus labels
            leg2 = ax.legend(line_handles, legend2_labels,
                             title="Nucleus@60 Elements",
                             bbox_to_anchor=(1.05, 0), # Place at the bottom-right
                             loc='lower left', 
                             fontsize='medium')
            
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Save the figure, adjusting layout to fit the legend
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Successfully saved per-sentence plots to {output_pdf}")
    
if __name__ == "__main__":
    all_data = main()
    for i, data in enumerate(all_data):
        plot_from_dict_list(data, f'mistral_example_trace_{i}.pdf')
