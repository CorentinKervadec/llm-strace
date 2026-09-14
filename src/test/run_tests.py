import subprocess

AVAILABLE_MODELS = [
    'Qwen/Qwen3-8B-Base',
    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2.5-7B"
]

for model in AVAILABLE_MODELS:
    print(f"\n{'='*50}\nTesting Model: {model}\n{'='*50}")
    # Call pytest for a specific model using standard subprocess (safe for CUDA)
    cmd = [
        "pytest", 
        "src/test/test_models.py", 
        "-k", model.split('/')[-1] # Only run tests matching this model name
    ]
    
    # Run the process and wait for it to finish. 
    # VRAM is 100% cleared by the OS when this returns.
    subprocess.run(cmd)