import subprocess
import time
import datetime
import os
import random
import re
import json
import sys
from collections import defaultdict

# --- Configuration ---
PROMPTS_FILE = "data/meta_prompts_v2.csv"
IMPORTANCE_MODE = "norm"
STRACE_MODE = "threshold"
START_STAGE = 1
END_STAGE = 3
NUM_SEEDS = 1

# -- Throttling & Priority Settings --
SEED_SUBMISSION_DELAY = 60    # Seconds to wait between seeds (avoids rapid-fire submission)
MODEL_SUBMISSION_DELAY = 5    # Seconds to wait between switching models
SLURM_NICE_VALUE = "0"     # Higher value = LOWER priority. (Standard user range usually 0-10000)

# -- File Paths --
LOG_FILE = "multi_model_experiment_6.log"
STATE_FILE = "experiment_state_6.json" # To save progress in case of crash

AVAILABLE_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    'allenai/OLMo-2-0425-1B',
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-7B",
]
SHORT_L=20
MEDIUM_L=30
LONG_L=30
LENGTH_PER_MODEL = {
    "Qwen/Qwen2-0.5B": LONG_L,
    "Qwen/Qwen2-1.5B": LONG_L,
    "Qwen/Qwen2-7B": MEDIUM_L,
    "Qwen/Qwen2.5-0.5B": LONG_L,
    "Qwen/Qwen2.5-1.5B": LONG_L,
    "Qwen/Qwen2.5-14B": SHORT_L,
    "Qwen/Qwen2.5-3B": LONG_L,
    "Qwen/Qwen2.5-7B": MEDIUM_L,
    "Qwen/Qwen3-0.6B-Base": LONG_L,
    "Qwen/Qwen3-1.7B-Base": LONG_L,
    "Qwen/Qwen3-4B-Base": MEDIUM_L,
    "Qwen/Qwen3-8B-Base": MEDIUM_L,
    "allenai/OLMo-2-0425-1B": LONG_L,
    "allenai/OLMo-2-1124-13B": SHORT_L,
    "allenai/OLMo-2-1124-7B": MEDIUM_L,
    "mistralai/Mistral-7B-v0.1": MEDIUM_L,
}

# --- Helper Functions ---

def log(message, print_to_console=True):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    if print_to_console:
        print(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def save_state(job_registry):
    """Saves the current job registry to a JSON file for recovery."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(job_registry, f, indent=4)
    except Exception as e:
        log(f"WARNING: Failed to save state file: {e}")

def load_state():
    """Loads previous job registry if it exists."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log(f"WARNING: Found state file but failed to load: {e}")
    return {}

def submit_job(model, seed):
    cmd = [
        "./GEN_MASTER_SLURM.sh",
        model,
        PROMPTS_FILE,
        IMPORTANCE_MODE,
        STRACE_MODE,
        str(LENGTH_PER_MODEL.get(model, MEDIUM_L)), # Added .get() safety
        str(seed),
        str(START_STAGE),
        str(END_STAGE)
    ]
    
    env = os.environ.copy()
    if SLURM_NICE_VALUE:
        env["SBATCH_NICE"] = SLURM_NICE_VALUE

    MAX_RETRIES = 100
    BASE_WAIT = 60
    LONG_WAIT = BASE_WAIT * 60

    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        
        # We initialize output vars to ensure scope availability
        output = ""
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            output = result.stdout.strip()
            # If we reach here, return code was 0 (clean success)

        except subprocess.CalledProcessError as e:
            # --- THE FIX ---
            # The script failed (non-zero exit), BUT it might have already printed the Job ID.
            # We must check stdout before deciding to retry.
            output = e.stdout.strip() if e.stdout else ""
            err_msg = e.stderr.strip() if e.stderr else "No stderr"
            
            # Check if we actually got IDs despite the crash
            if "Job ID" in output:
                log(f"WARNING: Script exited with error {e.returncode} but Job IDs were found. Treating as SUCCESS.")
                log(f"  (Ignored Error: {err_msg})")
                # We do NOT continue loop; we fall through to the parsing logic below
            else:
                # Genuine failure (no Job ID found). Handle retries/timeouts.
                elapsed = time.time() - t0
                
                is_congestion = any(x in err_msg for x in [
                    "temporarily unavailable",
                    "Socket timed out",
                    "temporarily unable to accept job",
                    "Connection refused",
                    "due to system load"
                ])
                
                if is_congestion:
                    wait_time = LONG_WAIT if elapsed > 60 else BASE_WAIT * (attempt + 1)
                    log(f"Slurm Congestion detected (Time: {int(elapsed)}s). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue # Retry loop
                else:
                    # Fatal error (e.g. invalid arguments), unlikely to be fixed by retrying
                    log(f"ERROR: Fatal error submitting {model} (Seed {seed})")
                    log(f"  Stderr: {err_msg}")
                    return None

        # --- PARSING LOGIC (Shared for Clean Success AND recovered 'Failed' Success) ---
        job_ids = re.findall(r'Job ID:\s*(\d+)', output)
        
        if job_ids:
            job_map = {}
            current_stage = START_STAGE
            for jid in job_ids:
                if current_stage <= END_STAGE:
                    job_map[str(current_stage)] = jid
                    current_stage += 1
            
            stages_str = ", ".join([f"S{s}:{jid}" for s, jid in job_map.items()])
            log(f"SUCCESS: Submitted {model} (Seed {seed}) -> {stages_str}")
            return job_map
        else:
            # If check=True passed (exit 0) but regex found nothing, something is weird.
            log(f"WARNING: Script finished successfully but parsing JobID failed. Output:\n{output}")
            return None

    log(f"CRITICAL: Failed to submit {model} after {MAX_RETRIES} attempts. Skipping.")
    return None

# def submit_job(model, seed):
#     """
#     Submits the GEN_MASTER_SLURM.sh script.
#     """
#     cmd = [
#         "./GEN_MASTER_SLURM.sh",
#         model,
#         PROMPTS_FILE,
#         IMPORTANCE_MODE,
#         STRACE_MODE,
#         str(LENGTH_PER_MODEL[model]),
#         str(seed),
#         str(START_STAGE),
#         str(END_STAGE)
#     ]
    
#     # Inject Nice value into environment for this subprocess
#     env = os.environ.copy()
#     if SLURM_NICE_VALUE:
#         env["SBATCH_NICE"] = SLURM_NICE_VALUE

#     try:
#         # Run subprocess with the modified environment
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
#         output = result.stdout.strip()
        
#         # Regex to capture Job ID.
#         job_ids = re.findall(r'Job ID:\s*(\d+)', output)
        
#         if job_ids:
#             job_map = {}
#             current_stage = START_STAGE
#             for jid in job_ids:
#                 if current_stage <= END_STAGE:
#                     job_map[str(current_stage)] = jid # Use string keys for JSON compatibility
#                     current_stage += 1
            
#             stages_str = ", ".join([f"S{s}:{jid}" for s, jid in job_map.items()])
#             log(f"SUCCESS: Submitted {model} (Seed {seed}) -> {stages_str}")
#             return job_map
#         else:
#             log(f"WARNING: Submitted {model} (Seed {seed}) but parsing JobID failed. Output:\n{output}")
#             return None

#     except subprocess.CalledProcessError as e:
#         log(f"ERROR: Failed to submit {model} (Seed {seed})")
#         log(f"  Stderr: {e.stderr}")
#         return None

def batch_list(iterable, n=1):
    """Yields successive n-sized chunks from iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_job_status_aggregate(base_job_ids):
    """
    Queries sacct for a list of base Job IDs. 
    Batches queries to avoid 'Argument list too long' errors.
    """
    if not base_job_ids:
        return {}
    
    aggregated_info = {}
    valid_ids = [str(jid) for jid in base_job_ids if jid]
    
    # Process in chunks of 100 to prevent command line overflow
    for chunk in batch_list(valid_ids, 100):
        job_id_str = ",".join(chunk)
        cmd = ['sacct', '-j', job_id_str, '-n', '-o', 'JobIDRaw,State,Elapsed']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            tasks_per_job = defaultdict(list)
            
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    jid_raw = parts[0]
                    state = parts[1]
                    elapsed = parts[2] if len(parts) > 2 else "00:00:00"
                    
                    if '.' in jid_raw: continue # Skip batch steps
                    
                    # Handle Array IDs: 12345_1 -> Base 12345
                    base_id = jid_raw.split('_')[0]
                    
                    if base_id in chunk:
                        tasks_per_job[base_id].append((state, elapsed))

            # Aggregate
            for base_id in chunk:
                tasks = tasks_per_job.get(base_id, [])
                if not tasks:
                    aggregated_info[base_id] = {'state': 'PENDING', 'elapsed': '00:00:00'}
                    continue
                    
                states = [t[0] for t in tasks]
                elapsed_times = [t[1] for t in tasks]
                
                # Simplified state logic
                if any(s.startswith('RUNNING') for s in states): final_state = 'RUNNING'
                elif any(s.startswith('PENDING') for s in states): final_state = 'PENDING'
                elif any(s.startswith('FAILED') or s.startswith('TIMEOUT') for s in states): final_state = 'FAILED'
                elif all(s.startswith('COMPLETED') for s in states): final_state = 'COMPLETED'
                elif any(s.startswith('CANCELLED') for s in states): final_state = 'CANCELLED'
                else: final_state = states[0]
                
                aggregated_info[base_id] = {'state': final_state, 'elapsed': elapsed_times[0]}

        except Exception as e:
            log(f"Warning: Failed to query sacct batch: {e}")

    return aggregated_info

def monitor_jobs(job_registry):
    """
    Loops until all submitted jobs are finished.
    """
    log("--- Starting Monitoring Phase ---")
    log("Tip: You can safely Ctrl+C this script; the job state is saved to experiment_state.json")
    
    active_jobs = set(job_registry.keys())
    
    try:
        while active_jobs:
            # Query status
            job_info = get_job_status_aggregate(list(active_jobs))
            
            current_active = set()
            completed_in_this_loop = 0
            
            for jid in active_jobs:
                if jid not in job_info:
                    current_active.add(jid) # Assume pending/unknown, keep tracking
                    continue
                    
                state = job_info[jid]['state']
                
                if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL']:
                    # Job is terminal
                    completed_in_this_loop += 1
                else:
                    current_active.add(jid)

            active_jobs = current_active
            
            # log status update
            log(f"Status Update: {len(active_jobs)} arrays still active. {completed_in_this_loop} finished recently.", print_to_console=True)
            
            if not active_jobs:
                break
                
            time.sleep(300) # Check every 5 minutes

    except KeyboardInterrupt:
        log("\n--- Monitoring Interrupted by User ---")
        log(f"State saved to {STATE_FILE}. You can restart monitoring later.")
        save_state(job_registry)
        sys.exit(0)

def main():
    log("=== Starting Multi-Model Experiment Run ===")
    
    # 1. Load existing state if we are resuming
    job_registry = load_state()
    if job_registry:
        log(f"Resumed from previous state file. Found {len(job_registry)} jobs already tracked.")
        # Ask user if they want to submit new ones or just monitor
        # For automation, we assume if state exists, we might still want to submit MISSING models? 
        # For now, let's assume we proceed to submit, but you might want logic to skip already done ones.
        pass

    start_time = time.time()

    # 2. Submission Phase
    # We loop through models/seeds. If we want to be smart, we check if they are already in job_registry.
    # To keep it simple, I assume you manage the lists or clear the json for a fresh run.
    
    log(f"--- Submission Phase (Delay: {SEED_SUBMISSION_DELAY}s per seed) ---")
    
    for model in AVAILABLE_MODELS:
        log(f"Processing Model: {model}")
        
        seeds = [random.randint(10000, 99999) for _ in range(NUM_SEEDS)]
        
        for i, seed in enumerate(seeds):
            # Generate a unique key to check if we already did this (optional safety)
            # submission_key = f"{model}_{seed}" 
            
            job_map = submit_job(model, seed)
            
            if job_map:
                for stage, jid in job_map.items():
                    job_registry[jid] = {
                        'model': model,
                        'seed': seed,
                        'stage': stage,
                        'submitted_at': time.time()
                    }
                # Save state immediately after successful submission
                save_state(job_registry)
            
            # Delay between seeds to prevent flooding
            time.sleep(SEED_SUBMISSION_DELAY)

        # Delay between models
        time.sleep(MODEL_SUBMISSION_DELAY)

    submission_end = time.time()
    log(f"=== All Submissions Complete ({len(job_registry)} jobs tracked) ===")
    
    # 3. Monitoring Phase
    if job_registry:
        monitor_jobs(job_registry)
    
    total_duration = time.time() - start_time
    log(f"=== Experiment Completed ===")
    log(f"Total Duration: {str(datetime.timedelta(seconds=int(total_duration)))}")

if __name__ == "__main__":
    main()