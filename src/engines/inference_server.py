import time
import uuid
import multiprocessing as mp
from queue import Empty
from typing import Dict, List, Any, Optional
import threading
import torch
import numpy as np
from torch.amp.autocast_mode import autocast

from searchless_chess.src.models.transformer import ChessTransformer
from searchless_chess.src import tokenizer


class InferenceRequest:
    def __init__(self, request_id: str, fens: List[str], worker_id: int):
        self.request_id = request_id
        self.fens = fens
        self.worker_id = worker_id
        self.timestamp = time.time()


class InferenceResponse:
    def __init__(self, request_id: str, results: Optional[Dict[str, np.ndarray]] = None, error: Optional[str] = None):
        self.request_id = request_id
        # Convert tensors to numpy arrays for workers to use directly
        if results is not None:
            self.results = {}
            for key, tensor in results.items():
                # Convert to float32 first (numpy doesn't support bfloat16), then to numpy array on CPU
                self.results[key] = tensor.float().detach().cpu().numpy()
        else:
            self.results = None
        self.error = error


class InferenceServer:
    def __init__(
        self,
        checkpoint_path: str,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        max_batch_size: int = 32,
        batch_timeout_ms: int = 10,
    ):
        self.checkpoint_path = checkpoint_path
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.running = False
        
        # For tracking pending requests
        self.pending_requests: List[InferenceRequest] = []
        self.batch_start_time = None
        
    def _load_model(self):
        """Load the model once during server initialization."""
        print(f"Loading model from {self.checkpoint_path} on {self.device}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        self.model = ChessTransformer(config=model_config).to(self.device)
        
        if checkpoint['compiled']:
            self.model = torch.compile(self.model)
            
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print("Model loaded successfully")
        
    def _split_batch_output(self, output: Dict[str, torch.Tensor], request_fen_counts: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Split batched model output back to individual request outputs."""
        results = []
        start_idx = 0
        
        for fen_count in request_fen_counts:
            end_idx = start_idx + fen_count
            
            # Extract slice for this request and create new tensors
            request_output = {}
            for key, tensor in output.items():
                # Use .clone() to ensure we get a new tensor, not a view
                request_output[key] = tensor[start_idx:end_idx].clone()
            
            results.append(request_output)
            start_idx = end_idx
            
        return results
        
    def _process_batch(self, requests: List[InferenceRequest]) -> None:
        """Process a batch of inference requests."""
        if not requests:
            return
            
        try:
            # Collect all FENs from all requests
            all_fens = []
            request_fen_counts = []
            
            for request in requests:
                all_fens.extend(request.fens)
                request_fen_counts.append(len(request.fens))
            
            # Tokenize all FENs at once
            x = []
            for fen in all_fens:
                x.append(tokenizer.tokenize(fen))
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            
            # Run inference
            with torch.inference_mode(), autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                output = self.model(x)
            
            # Split results back to individual requests
            split_results = self._split_batch_output(output, request_fen_counts)
            
            for i, request in enumerate(requests):
                response = InferenceResponse(
                    request_id=request.request_id,
                    results=split_results[i]
                )
                self.response_queue.put(response)
                
        except Exception as e:
            # Send error responses to all requests in the batch
            error_msg = f"Inference error: {str(e)}"
            print(f"Error in batch processing: {error_msg}")  # Debug print
            import traceback
            traceback.print_exc()  # Debug traceback
            for request in requests:
                response = InferenceResponse(
                    request_id=request.request_id,
                    results=None,
                    error=error_msg
                )
                self.response_queue.put(response)
    
    def _should_process_batch(self) -> bool:
        """Determine if we should process the current batch."""
        if not self.pending_requests:
            return False
            
        # Process if we hit max batch size
        if len(self.pending_requests) >= self.max_batch_size:
            return True
            
        # Process if timeout exceeded
        if self.batch_start_time is not None:
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            if elapsed_ms >= self.batch_timeout_ms:
                return True
                
        return False
    
    def run(self):
        """Main server loop."""
        print("Starting inference server...")
        self._load_model()
        self.running = True
        
        while self.running:
            try:
                # Try to get a request with a short timeout
                try:
                    request_data = self.request_queue.get(timeout=0.001)
                    if request_data is None:  # Shutdown signal
                        break
                        
                    request = InferenceRequest(**request_data)
                    
                    # Add to pending batch
                    if not self.pending_requests:
                        self.batch_start_time = time.time()
                    self.pending_requests.append(request)
                    
                except Empty:
                    pass
                
                # Check if we should process the current batch
                if self._should_process_batch():
                    self._process_batch(self.pending_requests)
                    self.pending_requests = []
                    self.batch_start_time = None
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in inference server: {e}")
                
        # Process any remaining requests
        if self.pending_requests:
            self._process_batch(self.pending_requests)
            
        print("Inference server shutting down...")


def start_inference_server(
    checkpoint_path: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    max_batch_size: int = 32,
    batch_timeout_ms: int = 10,
):
    """Function to run the inference server in a separate process."""
    server = InferenceServer(
        checkpoint_path=checkpoint_path,
        request_queue=request_queue,
        response_queue=response_queue,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
    )
    server.run() 