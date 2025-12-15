import subprocess
import time
import datetime
import os
import random
import re

# --- Configuration ---
PROMPTS_FILE = "data/prompts_test.txt"
IMPORTANCE_MODE = "norm"
STRACE_MODE = "threshold"
MAX_NEW_TOKENS = 20
START_STAGE = 1
END_STAGE = 3
NUM_SEEDS = 2

AVAILABLE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    'allenai/OLMo-2-0425-1B',
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
    # "google/gemma-3-270m",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-7B",
]

LOG_FILE = "multi_model_experiment.log"

# --- Helper Functions ---

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def submit_job(model, seed):
    """
    Submits the GEN_MASTER_SLURM.sh script for a specific model and seed.
    Returns a dictionary mapping Stage (1,2,3) to JobID if successful, None otherwise.
    """
    cmd = [
        "./GEN_MASTER_SLURM.sh",
        model,
        PROMPTS_FILE,
        IMPORTANCE_MODE,
        STRACE_MODE,
        str(MAX_NEW_TOKENS),
        str(seed),
        str(START_STAGE),
        str(END_STAGE)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        # Parse output for lines like "  -> Job ID: 12345"
        # We assume the order is sequential: Stage 1, Stage 2, Stage 3
        # GEN_MASTER_SLURM.sh prints them in order.
        
        job_ids = re.findall(r'Job ID: (\d+)', output)
        
        if job_ids:
            # Map found IDs to stages based on how many were submitted
            # This assumes standard execution order: 1 -> 2 -> 3
            # If start_stage > 1, the first ID found corresponds to that start stage.
            
            job_map = {}
            current_stage = START_STAGE
            for jid in job_ids:
                if current_stage <= END_STAGE:
                    job_map[current_stage] = jid
                    current_stage += 1
            
            stages_str = ", ".join([f"S{s}:{jid}" for s, jid in job_map.items()])
            log(f"SUCCESS: Submitted {model} (Seed {seed}) -> {stages_str}")
            return job_map
        else:
            log(f"WARNING: Submitted {model} (Seed {seed}) but found no JobIDs. Output:\n{output}")
            return None

    except subprocess.CalledProcessError as e:
        log(f"ERROR: Failed to submit {model} (Seed {seed})")
        log(f"  Stderr: {e.stderr}")
        return None

def get_job_info(job_ids):
    """
    Queries sacct to get status and elapsed time for a list of jobs.
    Returns a dict: {job_id: {'state': state, 'elapsed': elapsed_time}}
    """
    if not job_ids:
        return {}
    
    # Filter valid IDs
    valid_ids = [str(jid) for jid in job_ids if jid]
    if not valid_ids:
        return {}

    job_id_str = ",".join(valid_ids)
    
    # Query sacct for State and Elapsed time
    cmd = ['sacct', '-j', job_id_str, '-n', '-o', 'JobID,State,Elapsed']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                jid = parts[0]
                if '.' in jid: continue # Skip batch steps
                
                state = parts[1]
                elapsed = parts[2]
                info[jid] = {'state': state, 'elapsed': elapsed}
        return info
    except Exception as e:
        log(f"Warning: Failed to query sacct: {e}")
        return {}

def monitor_jobs(job_registry):
    """
    Loops until all submitted jobs are finished.
    job_registry: dict mapping job_id -> { 'model': str, 'seed': int, 'stage': int }
    """
    log("--- Starting Monitoring Phase ---")
    
    active_jobs = set(job_registry.keys())
    total_jobs = len(active_jobs)
    start_monitor = time.time()
    
    # Track completion
    completed_jobs = set()

    while active_jobs:
        # Query status
        job_info = get_job_info(list(active_jobs))
        
        current_active = set()
        
        for jid in active_jobs:
            # Metadata for logging
            meta = job_registry[jid]
            label = f"{meta['model']} (Seed {meta['seed']}) Stage {meta['stage']}"

            if jid not in job_info:
                # Job might be queued but not in accounting yet, assume active
                current_active.add(jid)
                continue
                
            state = job_info[jid]['state']
            elapsed = job_info[jid]['elapsed']
            
            # Check terminal states
            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL']:
                if jid not in completed_jobs:
                    log(f"[{state}] Job {jid} finished: {label}. Duration: {elapsed}")
                    completed_jobs.add(jid)
            else:
                # PENDING, RUNNING, REQUEUED
                current_active.add(jid)

        active_jobs = current_active
        finished_count = len(completed_jobs)
        
        elapsed_total = time.time() - start_monitor
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_total)))
        
        log(f"Status: {finished_count}/{total_jobs} finished. {len(active_jobs)} active.")
        log(f"  Monitor Running Time: {elapsed_str}")
        
        if not active_jobs:
            break
            
        # Check every 5 minutes
        time.sleep(300) 

def main():
    log("=== Starting Multi-Model Experiment Run ===")
    
    start_time = time.time()
    
    # Registry stores metadata for every job ID we launch
    # Format: { job_id: { 'model': ..., 'seed': ..., 'stage': ... } }
    job_registry = {}

    # 1. Submission Phase
    for model in AVAILABLE_MODELS:
        log(f"--- Processing Model: {model} ---")
        
        seeds = [random.randint(10000, 99999) for _ in range(NUM_SEEDS)]
        
        for i, seed in enumerate(seeds):
            # submit_job returns a dict: {stage_num: job_id}
            job_map = submit_job(model, seed)
            
            if job_map:
                for stage, jid in job_map.items():
                    job_registry[jid] = {
                        'model': model,
                        'seed': seed,
                        'stage': stage
                    }
            
            time.sleep(1) # Short delay

    submission_end = time.time()
    log(f"=== All Submissions Complete ({len(job_registry)} individual jobs tracked) ===")
    log(f"Submission duration: {(submission_end - start_time):.2f}s")
    
    # 2. Monitoring Phase
    if job_registry:
        monitor_jobs(job_registry)
    
    total_duration = time.time() - start_time
    log(f"=== Experiment Completed ===")
    log(f"Total Duration: {str(datetime.timedelta(seconds=int(total_duration)))}")

if __name__ == "__main__":
    main()