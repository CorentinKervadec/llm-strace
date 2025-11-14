from src.llm_graph.llm_graph_nx import LLM_Graph_NX
from src.llm_hooked.olmo_hooked import Olmo2_Hooked
from typing import Optional

class Olmo2_Graph_NX(LLM_Graph_NX):
    """
    Represent Olmo2 LLM as a computational graph
    """
    def __init__(
            self,
            olmo_hooked: Optional[Olmo2_Hooked] = None, 
            input_sentence: Optional[str] = None, 
            importance_mode: Optional[str] = 'dist',
            **attr):
        super().__init__(olmo_hooked, input_sentence, importance_mode, **attr)
