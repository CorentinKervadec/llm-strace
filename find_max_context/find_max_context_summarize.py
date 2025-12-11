import glob
import re
import os

# Configuration: Point this to where your .txt logs are stored
# Use "." if they are in the current directory
LOG_DIRECTORY = "./context_logs" 

def parse_logs(log_directory):
    """
    Scans log files, extracts the real model name and max context length.
    """
    results = []
    
    # 1. Find all log files ending in _context_search.txt
    log_files = glob.glob(os.path.join(log_directory, "*_context_search.txt"))
    
    if not log_files:
        print(f"No log files found in directory: {log_directory}")
        return []

    # 2. Define Regex patterns
    # Pattern to find the final result number
    result_pattern = re.compile(r"FINAL RESULT: Maximum stable context length is\s+(\d+)")
    
    # Pattern to find the real model name from the file header
    # Matches: "=== Optimization Run for mistralai/Mistral-7B-v0.1 ==="
    name_pattern = re.compile(r"=== Optimization Run for (.+?) ===")

    print(f"Found {len(log_files)} log files. Parsing...\n")

    for log_file in log_files:
        model_name = "Unknown"
        max_len = "-"
        status = "Unknown"

        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # A. Extract Model Name (from header)
            name_match = name_pattern.search(content)
            if name_match:
                model_name = name_match.group(1).strip()
            else:
                # Fallback: Extract from filename if header is missing
                base = os.path.basename(log_file)
                model_name = base.replace("_context_search.txt", "")

            # B. Extract Result
            result_match = result_pattern.search(content)
            
            if result_match:
                max_len = int(result_match.group(1))
                status = "Done"
            else:
                # Check why it's not done
                if "System Error" in content:
                    status = "Error/Crashed"
                elif "Job Failed" in content and "Exit Code" in content:
                    # Finds the last exit code to hint at what happened
                    status = "Failed (Last Job)"
                else:
                    status = "Running..."

            results.append((model_name, max_len, status))

        except Exception as e:
            results.append((os.path.basename(log_file), "?", f"Read Error: {str(e)[:20]}"))

    # Sort alphabetically by model name
    results.sort(key=lambda x: x[0])
    
    return results

def print_table(results):
    """Prints a pretty Markdown-compatible table."""
    if not results:
        return

    # Calculate column widths dynamically
    # Minimum width of 25 for name, 15 for context
    col_model = max(max(len(str(r[0])) for r in results), 20) + 2
    col_ctx = 15
    col_status = 20
    
    # Create Header
    header = f"| {'Model Name'.ljust(col_model)} | {'Max Context'.center(col_ctx)} | {'Status'.center(col_status)} |"
    separator = f"|{'-'*(col_model+2)}|{'-'*(col_ctx+2)}|{'-'*(col_status+2)}|"
    
    print(separator)
    print(header)
    print(separator)
    
    for model, ctx, status in results:
        ctx_str = str(ctx)
        print(f"| {model.ljust(col_model)} | {ctx_str.center(col_ctx)} | {status.center(col_status)} |")
    
    print(separator)

if __name__ == "__main__":
    data = parse_logs(LOG_DIRECTORY)
    print_table(data)