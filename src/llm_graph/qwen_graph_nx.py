from src.llm_graph.llm_graph_nx import LLM_Graph_NX
from src.llm_hooked.qwen_hooked import Qwen_Hooked
from typing import Optional

class Qwen_Graph_NX(LLM_Graph_NX):
    """
    Represent Qwen LLM as a computational graph
    """
    def __init__(
            self,
            qwen_hooked: Optional[Qwen_Hooked] = None, 
            input_sentence: Optional[str] = None, 
            importance_mode: Optional[str] = 'dist',
            **attr):
        super().__init__(qwen_hooked, input_sentence, importance_mode, **attr)
