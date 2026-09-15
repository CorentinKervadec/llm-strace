import io
import sys
from contextlib import redirect_stdout

class SuppressUnitTestLogs:
    """
    Captures verbose stdout from unit tests, summarizing output to a single
    pass message or explicitly printing failed components.
    """
    def __enter__(self):
        self.buffer = io.StringIO()
        self._redirector = redirect_stdout(self.buffer)
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirector.__exit__(exc_type, exc_val, exc_tb)
        output = self.buffer.getvalue()
        
        # Split into non-empty lines
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return

        # Separate successful assertions from failures
        failures = [line for line in lines if "success!" not in line.lower()]
        
        if exc_type is None and not failures:
            print(f"[UNIT TEST] All {len(lines)} graph layer test passed successfully! ✅")
        else:
            print(f"[UNIT TEST] ❌ Graph population encountered failures ({len(failures)} item(s) failed):")
            for fail in failures:
                print(f"  • {fail}")