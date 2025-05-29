"""Client-side inference manager for communicating with InferenceServer."""

import multiprocessing as mp
import uuid
import time
import queue
from typing import Dict, Any, Optional
import chess

from .server import InferenceRequest, InferenceResponse


class InferenceManager:
    """
    Client-side manager for requesting inference from the remote InferenceServer.
    Provides a clean interface that search algorithms can use.
    """
    
    def __init__(self, request_queue: mp.Queue, timeout: float = 10.0):
        self.request_queue = request_queue
        self.timeout = timeout
        # Create response queue in this process
        self.response_queue = mp.Queue()
        
    def request_inference(self, board: chess.Board) -> Dict[str, Any]:
        """
        Request inference for a single board position.
        
        Args:
            board: Chess board to analyze
            
        Returns:
            Model output dictionary
            
        Raises:
            TimeoutError: If inference request times out
            RuntimeError: If inference fails
        """
        return self.request_inference_fen(board.fen())
    
    def request_inference_fen(self, fen: str) -> Dict[str, Any]:
        """
        Request inference for a FEN string.
        
        Args:
            fen: FEN string of position to analyze
            
        Returns:
            Model output dictionary
        """
        request_id = str(uuid.uuid4())
        request = InferenceRequest(request_id, fen, self.response_queue)
        
        try:
            # Send request to inference server
            self.request_queue.put(request, timeout=1.0)
        except queue.Full:
            raise RuntimeError("Inference request queue is full")
        
        # Wait for response
        try:
            response = self.response_queue.get(timeout=self.timeout)
            
            if response.error:
                raise RuntimeError(f"Inference failed: {response.error}")
            
            return response.output
            
        except queue.Empty:
            raise TimeoutError(f"Inference request timed out after {self.timeout}s")
    
    def request_batch_inference(self, boards: list[chess.Board]) -> list[Dict[str, Any]]:
        """
        Request inference for multiple boards.
        
        Args:
            boards: List of chess boards to analyze
            
        Returns:
            List of model output dictionaries
        """
        fens = [board.fen() for board in boards]
        return self.request_batch_inference_fens(fens)
    
    def request_batch_inference_fens(self, fens: list[str]) -> list[Dict[str, Any]]:
        """
        Request inference for multiple FEN strings.
        
        Args:
            fens: List of FEN strings to analyze
            
        Returns:
            List of model output dictionaries
        """
        requests = []
        request_ids = []
        
        # Send all requests
        for fen in fens:
            request_id = str(uuid.uuid4())
            request = InferenceRequest(request_id, fen, self.response_queue)
            requests.append(request)
            request_ids.append(request_id)
            
            try:
                self.request_queue.put(request, timeout=1.0)
            except queue.Full:
                raise RuntimeError("Inference request queue is full")
        
        # Collect all responses
        responses = {}
        for _ in range(len(requests)):
            try:
                response = self.response_queue.get(timeout=self.timeout)
                
                if response.error:
                    raise RuntimeError(f"Inference failed: {response.error}")
                
                responses[response.request_id] = response.output
                
            except queue.Empty:
                raise TimeoutError(f"Batch inference request timed out after {self.timeout}s")
        
        # Return responses in the same order as requests
        return [responses[request_id] for request_id in request_ids] 