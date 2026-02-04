from src.llm_trace.llm_trace import load_from_file_light
from src.llm_hooked.hook_constructors import get_hooked_constructor
import os
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-8B-Base"
PATH_RESULTS = "./results_norm_threshold"
FOLDER_RESULTS = "final_straces_20"


HOOKED_CONSTRUCT = get_hooked_constructor(MODEL_NAME)
llm_hooked = HOOKED_CONSTRUCT(MODEL_NAME, half_precision=True, untrained=False)
llm_hooked.remove_extraction_hooks()

split_name = MODEL_NAME.split('/')[-1]
path = f'{PATH_RESULTS}/{split_name}/{FOLDER_RESULTS}'

for filename in sorted(os.listdir(path)):
    if filename.startswith('strace_final_') and filename.endswith('.npz'):
        filepath = os.path.join(path, filename)
        # Process the file here
        strace = load_from_file_light(filepath, llm_hooked)
        strace.print_graph_sizes_and_thresholds_short()

        input("\n\n----------- NEXT?")