# src/modified_transformers/unit_tests.py
from typing import Literal
import torch

STRICT_TOLERANCE = 1e-5
LOOSE_TOLERANCE = 5e-2

# --- Custom Exception Classes ---
class PrecisionError(AssertionError):
    """Raised when reconstruction fails due to floating point drift or reconstruction issue."""
    pass

class TopologyError(AssertionError):
    """Raised when the graph dimensions or directed acyclic nature is violated."""
    pass

class CausalLeakageError(AssertionError):
    """Raised when a token attends to future tokens."""
    pass

class MaskingError(AssertionError):
    """Raised when the masking provide inconsistent results."""
    pass

def test_reconstruction(original_vector, reconstructed_vector, message: str = None, tolerance: Literal['strict', 'loose'] = None, atol=None):
    if tolerance is None and atol is not None:
        current_atol = atol
    else:
        if tolerance not in ('strict', 'loose'):
            raise ValueError(f"Argument 'tolerance' must be 'strict' or 'loose'. Got: '{tolerance}'")
        current_atol = STRICT_TOLERANCE if tolerance == 'strict' else LOOSE_TOLERANCE

    test = torch.allclose(original_vector, reconstructed_vector, atol=current_atol)
    
    max_diff = (original_vector - reconstructed_vector).abs().max().item()

    if message:
        str_output = "success!" if test else f"failed. Max diff: {max_diff:.6f}"
        print(f"{message}: {str_output}")

    if not test:
        # Include the max_diff in the error message so the report generator can parse it!
        raise PrecisionError(f"[{message}] Failed. Expected atol={current_atol}, got max_diff={max_diff:.6f}")

    return test