import argparse
import sys
import html
import re
from transformers import AutoTokenizer
import pandas as pd

def generate_html_report(df, tokenizer, prompts_df, output_file="colored_output.html"):
    
    # --- Configuration ---
    # Bins for AUC (0-1 range)
    auc_bins = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    auc_colors = [
        "#440154", "#3b528b", "#21918c", "#5ec962", 
        "#fde725", "#fee08b", "#fdae61", "#d90429"
    ]
    
    # Bins for Entropy
    # We use the same color palette (Viridis/Magma) for visual consistency,
    # but map them to typical entropy ranges.
    ent_bins = [1., 2., 3., 4., 5., 6., 7., 8.] 
    
    # --- Start Building HTML ---
    html_content = []
    html_content.append("<html><head><meta charset='utf-8'><title>Interactive Gen Visualization</title>")
    html_content.append("""
    <style>
        body { font-family: 'Courier New', monospace; padding: 20px; background-color: #f4f4f4; }
        .controls {
            position: sticky; top: 0; 
            background: white; padding: 15px; 
            border-bottom: 2px solid #ddd; z-index: 1000;
            margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            display: flex; align_items: center; gap: 20px; flex-wrap: wrap;
        }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        .radio-group { display: flex; gap: 10px; align-items: center; }
        
        .prompt-block { 
            background: white; padding: 15px; margin-bottom: 20px; 
            border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            font-size: 16px; display: block; line-height: 1.6;
        }
        .prefix { color: black; font-weight: normal; white-space: pre-wrap; }
        .token { 
            white-space: pre-wrap; cursor: help; 
            border-radius: 2px; display: inline;
            transition: color 0.1s ease;
        }
        .legend-item {
            display: inline-block; margin-right: 8px; margin-bottom: 5px;
            font-size: 12px; background: #eee; border-radius: 3px;
        }
        .legend-color {
            display: inline-block; width: 15px; height: 15px;
            vertical-align: middle; border-radius: 2px 0 0 2px; margin-right: 5px;
        }
        .legend-text { padding-right: 5px; vertical-align: middle; }
        h3 { margin-top: 0; color: #333; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
    </style>
    </head><body>""")

    # --- Javascript ---
    html_content.append(f"""
    <script>
        // Store config for both metrics
        const CONFIG = {{
            'auc': {{
                'bins': {auc_bins},
                'colors': {auc_colors}
            }},
            'entropy': {{
                'bins': {ent_bins},
                'colors': {auc_colors}
            }}
        }};

        let currentMetric = 'auc';

        function getColor(value, bins, colors) {{
            value = Math.max(0, value);
            let idx = 0;
            while (idx < bins.length && value > bins[idx]) {{
                idx++;
            }}
            if (idx >= bins.length) idx = bins.length - 1;
            return colors[idx];
        }}

        function updateView() {{
            const slider = document.getElementById('contrastSlider');
            const displayVal = document.getElementById('sliderVal');
            const gamma = parseFloat(slider.value);
            
            // Get selected metric
            const radios = document.getElementsByName('metric');
            for (let r of radios) {{ if (r.checked) currentMetric = r.value; }}

            displayVal.textContent = gamma.toFixed(2);
            
            const bins = CONFIG[currentMetric].bins;
            const colors = CONFIG[currentMetric].colors;

            // 1. Update Token Colors
            const tokens = document.querySelectorAll('.token');
            tokens.forEach(tok => {{
                let rawVal = parseFloat(tok.getAttribute('data-' + currentMetric));
                if (!isNaN(rawVal)) {{
                    const adjustedVal = Math.pow(rawVal, gamma);
                    tok.style.color = getColor(adjustedVal, bins, colors);
                }}
            }});

            // 2. Update Legend
            const legendContainer = document.getElementById('legend-container');
            legendContainer.innerHTML = ''; 
            
            for (let i = 0; i < bins.length; i++) {{
                const binLimit = bins[i];
                const effectiveLimit = Math.pow(binLimit, 1.0 / gamma);
                const color = colors[i];

                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <span class='legend-color' style='background-color:${{color}};'></span>
                    <span class='legend-text'>&le; ${{effectiveLimit.toFixed(3)}}</span>
                `;
                legendContainer.appendChild(item);
            }}
        }}

        window.onload = updateView;
    </script>
    """)

    # --- Controls UI ---
    html_content.append("""
    <div class="controls">
        <div class="control-group">
            <b>Metric:</b>
            <div class="radio-group">
                <label><input type="radio" name="metric" value="auc" checked onchange="updateView()"> AUC</label>
                <label><input type="radio" name="metric" value="entropy" onchange="updateView()"> Entropy</label>
            </div>
        </div>
        
        <div style="width: 1px; background: #ddd; height: 40px;"></div>

        <div class="control-group" style="flex-grow: 1; max-width: 300px;">
            <b>Contrast (Slider):</b>
            <div style="display: flex; align-items: center; gap: 10px;">
                <input type="range" id="contrastSlider" min="0.1" max="4.0" step="0.1" value="1.0" style="width: 100%;" oninput="updateView()">
                <div style="font-weight: bold; width: 40px;" id="sliderVal">1.00</div>
            </div>
        </div>

        <div style="width: 1px; background: #ddd; height: 40px;"></div>

        <div class="control-group">
            <b>Legend (Thresholds):</b>
            <div id="legend-container"></div>
        </div>
    </div>
    """)

    # --- 2. Process Data ---
    if 'seed' in df.columns:
        grouper = ['prompt_name', 'seed']
    else:
        grouper = ['prompt_name']

    grouped = df.groupby(grouper)

    for name, group in grouped:
        if isinstance(name, tuple):
            prompt_id_str = str(name[0])
            display_name = f"{name[0]} (Seed: {name[1]})"
        else:
            prompt_id_str = str(name)
            display_name = name

        # --- Extract Prompt Prefix ---
        prompt_prefix_text = "[Prefix not found]"
        try:
            if "_" in prompt_id_str:
                idx = int(prompt_id_str.split('_')[-1])
                if 0 <= idx < len(prompts_df):
                    prompt_prefix_text = prompts_df.iloc[idx]['prompt']
                else:
                    prompt_prefix_text = f"[Index {idx} out of bounds]"
            else:
                 prompt_prefix_text = "[Invalid format]"
        except Exception:
            pass

        group = group.sort_values('step')
        group = group[(group['step'] >= 0) & (group['step'] <= 29)]

        html_content.append(f"<div class='prompt-block'><h3>{display_name}</h3><div>")
        safe_prefix = html.escape(str(prompt_prefix_text))
        html_content.append(f"<span class='prefix'>{safe_prefix}</span>")

        for _, row in group.iterrows():
            token_id = int(row['next_token_id'])
            auc_val = row.get('auc_tv', 0.0)
            ent_val = row.get('gen_entropy', 0.0)
            prob_val = row.get('next_token_prob', 0.0)
            
            # --- Decode Main Token (Robust) ---
            decoded_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if not decoded_text.startswith(" "):
                raw_token = tokenizer.convert_ids_to_tokens(token_id)
                if raw_token and (raw_token.startswith('\u2581') or raw_token.startswith('Ġ')):
                    decoded_text = " " + decoded_text
            safe_text = html.escape(decoded_text).replace(' ', '&nbsp;')
            
            # --- Decode Alternatives (RESTORED LOGIC) ---
            raw_alts = str(row.get('nucleus_token_id', ''))
            tooltip_alts = []
            
            if raw_alts and raw_alts.lower() != 'nan':
                # Use regex to find all numbers regardless of formatting (brackets, commas, etc)
                alt_ids = re.findall(r'\d+', raw_alts)
                
                for aid_str in alt_ids:
                    try:
                        aid = int(aid_str)
                        alt_dec = tokenizer.decode([aid], clean_up_tokenization_spaces=False)
                        
                        # Apply same space fix to alternatives
                        raw_alt_tok = tokenizer.convert_ids_to_tokens(aid)
                        if raw_alt_tok and not alt_dec.startswith(" ") and (raw_alt_tok.startswith('\u2581') or raw_alt_tok.startswith('Ġ')):
                            alt_dec = " " + alt_dec

                        # Escape for HTML attribute (newlines become \n text)
                        safe_alt = html.escape(alt_dec).replace('\n', '\\n')
                        tooltip_alts.append(f"{safe_alt}")
                    except Exception:
                        pass

            # --- Construct Full Tooltip ---
            tooltip_lines = [
                f"Step: {row['step']}",
                f"ID: {token_id}",
                f"AUC: {auc_val:.4f}",
                f"Entropy: {ent_val:.4f}",
                f"Prob: {prob_val:.4f}",
                "--- Alternatives ---"
            ] + tooltip_alts
            
            tooltip_str = "&#10;".join(tooltip_lines)
            
            # Write Span
            html_content.append(f"<span class='token' data-auc='{auc_val}' data-entropy='{ent_val}' title='{tooltip_str}'>{safe_text}</span>")

        html_content.append("</div></div>")

    html_content.append("</body></html>")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(html_content))
        print(f"HTML output written to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Input TSV")
    parser.add_argument('--prompt_file', type=str, required=True, help="Input CSV with prompt")
    parser.add_argument('--output', type=str, default="colored_output.html")
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        sys.exit(1)

    try:
        df = pd.read_csv(args.data, sep='\t')
        prompts_df = pd.read_csv(args.prompt_file)
        generate_html_report(df, tokenizer, prompts_df, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()