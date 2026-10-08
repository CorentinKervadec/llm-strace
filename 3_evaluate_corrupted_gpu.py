import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import cpu_offload

# Internal project imports
from src.llm_trace.llm_trace import load_from_file_light
from src.modified_transformers.utils import get_model_class, identify_model_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate s-trace context sensitivity under corrupted input."
    )
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="main",
        help="Model revision/checkpoint branch.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Model numerical precision.",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload for large models.",
    )

    # Dataset & directory paths
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV containing corrupted inputs.",
    )
    parser.add_argument(
        "--strace_dir",
        type=str,
        required=True,
        help="Directory containing saved clean strace_{i}.npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save corrupted evaluation results.",
    )

    # Chunking / Distributed execution parameters
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="Chunk ID for distributed SLURM job array processing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of sentences per chunk.",
    )
    parser.add_argument(
        "--total_sentences",
        type=int,
        required=True,
        help="Total size of the dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Model & Tokenizer Initialization
    # -------------------------------------------------------------------------
    print(f"[STAGE 3 | CHUNK {args.chunk_id}] Initializing model {args.model_name}...")
    start_time = time.time()

    target_dtype = torch.float16 if args.precision == "float16" else torch.float32
    model_type = identify_model_type(args.model_name)
    hf_constructor = get_model_class(model_type)

    if args.cpu_offload:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint,
        )
        llm = cpu_offload(llm, execution_device="cuda:0")
    else:
        llm = hf_constructor.from_pretrained(
            args.model_name,
            device_map="auto",
            attn_implementation="eager",
            use_safetensors=True,
            torch_dtype=target_dtype,
            revision=args.checkpoint,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[STAGE 3] {args.model_name} loaded in {time.time() - start_time:.2f} s")

    # -------------------------------------------------------------------------
    # 2. Dataset & Chunk Range Setup
    # -------------------------------------------------------------------------
    print(f"[STAGE 3] Loading corrupted dataset from [{args.csv_path}]")
    df = pd.read_csv(args.csv_path)

    # Determine sentence index range for this array chunk
    start_index = args.chunk_id * args.chunk_size
    end_index = min((args.chunk_id + 1) * args.chunk_size, args.total_sentences)

    print(
        f"[STAGE 3 | CHUNK {args.chunk_id}] Processing dataset subset: "
        f"Sentences {start_index} to {end_index - 1}"
    )

    # -------------------------------------------------------------------------
    # 3. Corrupted Context Evaluation Loop
    # -------------------------------------------------------------------------
    for i in range(start_index, end_index):
        strace_file = os.path.join(args.strace_dir, f"strace_final_{i}.npz")
        out_npz_path = os.path.join(args.output_dir, f"corrupted_strace_{i}.npz")

        # Ensure the prerequisite uncorrupted s-trace exists
        if not os.path.exists(strace_file):
            print(f"Warning: Clean trace file {strace_file} not found. Skipping.")
            continue

        # Extract corrupted row corresponding to sentence_id
        row = df[df["sentence_id"] == i]
        if row.empty:
            print(f"Warning: sentence_id {i} not found in CSV. Skipping.")
            continue

        corrupted_text = row["new_text"].values[0]
        next_word = row["next_word"].values[0] if "next_word" in row.columns else ""

        # Load clean trace masks and original graph metadata
        strace = load_from_file_light(strace_file, llm, tokenizer)

        # Run masked forward passes on corrupted context and save metrics
        # try:
        strace.compute_corrupted_stratum_evaluation(corrupted_text, next_word)
        strace.save_light_corrupted(out_npz_path)

        print(
            f"Successfully processed and saved sentence_id={i} to {out_npz_path}"
        )

        # except AssertionError as e:
        #     print(f"Error for sentence_id={i}: {e}")
        # except Exception as e:
        #     print(f"Unexpected error for sentence_id={i}: {e}")


if __name__ == "__main__":
    main()