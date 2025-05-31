from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass
import chess


@dataclass
class SearchResult:
    """Result of a search algorithm containing the best move, score, and metadata."""
    move: Optional[chess.Move]
    score: float
    metadata: Dict[str, Any]


class SearchAlgorithm(ABC):
    """Abstract base class for all search algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            'num_nodes': 0,
            'depth': 0,
        }
    
    @abstractmethod
    def search(
        self, 
        board: chess.Board, 
        inference_func: Callable[[chess.Board], Any],
        batch_inference_func: Optional[Callable] = None,
        depth: float = 2.0,
        **kwargs
    ) -> SearchResult:
        """
        Perform search on the given board position.
        
        Args:
            board: The chess board position to search from
            inference_func: Function that takes a board and returns model output
            batch_inference_func: Optional function for batch inference  
            depth: Search depth (can be fractional)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            SearchResult containing best move, score, and metadata
        """
        pass
    
    def reset_metrics(self):
        """Reset search metrics."""
        self.metrics = {
            'num_nodes': 0,
            'depth': 0,
        } 