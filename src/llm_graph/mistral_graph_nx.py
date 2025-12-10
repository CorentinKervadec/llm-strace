from src.llm_graph.llm_graph_nx import LLM_Graph_NX
from src.llm_hooked.mistral_hooked import Mistral_Hooked
from typing import Optional

class Mistral_Graph_NX(LLM_Graph_NX):
    """
    Represent Mistral LLM as a computational graph
    """
    def __init__(
            self,
            mistral_hooked: Optional[Mistral_Hooked] = None, 
            input_sentence: Optional[str] = None, 
            importance_mode: Optional[str] = None,
            **attr):
        super().__init__(mistral_hooked, input_sentence, importance_mode, **attr)
