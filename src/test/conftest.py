# ==========================================
# REPORT GENERATOR HOOK
# ==========================================
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generates a Markdown report at the end of the test suite."""
    
    with open("test_report.md", "w", encoding="utf-8") as f:
        f.write("# Mechanistic Interpretability Test Report\n\n")
        f.write(f"**Overall Status:** {'✅ PASSED' if exitstatus == 0 else '❌ FAILED'}\n\n")
        
        f.write("## Detailed Results\n")
        # Added 'Dataset' column to support our new wikitext loader
        f.write("| Model | Dataset | Dtype | Tokens | Time (s) | Result | Failure Cause | Details |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        # FIX 1: Explicitly gather reports from all possible status categories
        reports = []
        for status in ['passed', 'failed', 'skipped', 'error']:
            reports.extend(terminalreporter.getreports(status))

        # Iterate through all run tests
        for report in reports:
            # Skip setup/teardown phases, EXCEPT for skipped tests (which skip during setup)
            if report.when != 'call' and not (report.skipped and report.when == 'setup'):
                continue 
            
            # Extract recorded properties
            props = dict(report.user_properties)
            
            # FIX 2: Safely handle missing properties. 
            # If a test crashes early or is skipped, 'props' will be empty.
            model = props.get('model', 'Unknown')
            dataset = props.get('dataset', 'Unknown')
            dtype = props.get('dtype', 'Unknown')
            tokens = props.get('tokens', 'N/A')
            time_s = f"{props.get('time', 0.0):.3f}"
            
            # If properties are completely empty, try to extract basic info from the test node name
            if not props and "[" in report.nodeid:
                raw_params = report.nodeid.split("[")[-1].replace("]", "")
                model = f"*{raw_params}*" # Shows the raw parameterized string
            
            if report.passed:
                result = "✅ Pass"
                cause = "-"
                details = "Perfect reconstruction."
            elif report.skipped:
                result = "⏭️ Skip"
                cause = "Not Implemented / Skipped"
                # Extract the skip reason directly from Pytest
                details = str(report.longrepr[2]) if hasattr(report.longrepr, '__getitem__') else "Skipped"
            else:
                result = "❌ Fail"
                cause = props.get('failure_type', 'Unknown Failure')
                details = props.get('status_msg', 'No details provided.')
                
                # If a core dump or syntax error happened before the try/except block
                if cause == 'Unknown Failure' and hasattr(report, 'longreprtext'):
                    details = report.longreprtext.split('\n')[-1]
                    
            # Clean up newlines and markdown pipe characters for the table
            details = str(details).replace('\n', ' <br> ').replace('|', '\\|')

            f.write(f"| {model} | {dataset} | `{dtype}` | {tokens} | {time_s} | {result} | **{cause}** | {details} |\n")