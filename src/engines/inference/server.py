"""GPU inference server that runs in a separate process."""

import multiprocessing as mp
import time
import queue
from typing import List, Dict, Any, Optional
import threading
import uuid

import torch
import numpy as np
from torch.amp.autocast_mode import autocast

from searchless_chess.src import tokenizer
from searchless_chess.src.models.transformer import ChessTransformer


class InferenceRequest:
    """Request for model inference."""
    def __init__(self, request_id: str, fen: str):
        self.request_id = request_id
        self.fen = fen
        self.timestamp = time.time()


class InferenceResponse:
    """Response from model inference."""
    def __init__(self, request_id: str, output: Dict[str, Any], error: Optional[str] = None):
        self.request_id = request_id
        self.output = output
        self.error = error


class InferenceServer(mp.Process):
    """
    GPU inference server that runs in a separate process.
    Batches multiple inference requests for optimal GPU utilization.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        max_batch_size: int = 64,
        batch_timeout_ms: float = 10.0,
        device: Optional[str] = None
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms / 1000.0  # Convert to seconds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Will be initialized in the process
        self.model = None
        self.current_batch = []
        self.batch_timer = None
        self.shutdown_flag = False
        
    def run(self):
        """Main server loop - runs in the inference process."""
        try:
            self._initialize_model()
            self._server_loop()
        except Exception as e:
            print(f"InferenceServer error: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_model(self):
        """Initialize the model in the inference process."""
        print(f"InferenceServer: Loading model on {self.device}")
        
        # Set device and load model - handle CUDA in subprocess
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                print("InferenceServer: CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            device = torch.device(self.device)
            
            # Load checkpoint with proper device mapping
            if self.device == "cpu":
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            else:
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                
            model_config = checkpoint['model_config']
            
            self.model = ChessTransformer(config=model_config).to(device)
            
            if checkpoint.get('compiled', False):
                self.model = torch.compile(self.model)
            
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            
            # Set default tensor type
            torch.set_default_dtype(torch.float32)
            
            print(f"InferenceServer: Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"InferenceServer: Error loading model: {e}")
            # Try fallback to CPU if CUDA fails
            if self.device != "cpu":
                print("InferenceServer: Trying CPU fallback...")
                self.device = "cpu"
                device = torch.device("cpu")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                model_config = checkpoint['model_config']
                
                self.model = ChessTransformer(config=model_config).to(device)
                self.model.load_state_dict(checkpoint['model'])
                self.model.eval()
                torch.set_default_dtype(torch.float32)
                print("InferenceServer: CPU fallback successful")
            else:
                raise
    
    def _server_loop(self):
        """Main server loop that processes requests."""
        while not self.shutdown_flag:
            try:
                # Try to get a request (non-blocking with timeout)
                try:
                    request = self.request_queue.get(timeout=0.001)  # 1ms timeout
                    
                    if request == "SHUTDOWN":
                        self.shutdown_flag = True
                        break
                    
                    self._add_to_batch(request)
                    
                except queue.Empty:
                    # No new requests, check if we should process current batch
                    if self.current_batch and self._should_process_batch():
                        self._process_batch()
                    continue
                    
            except Exception as e:
                print(f"InferenceServer loop error: {e}")
                import traceback
                traceback.print_exc()
    
    def _add_to_batch(self, request: InferenceRequest):
        """Add request to current batch."""
        self.current_batch.append(request)
        
        # Start batch timer if this is the first request
        if len(self.current_batch) == 1:
            self.batch_start_time = time.time()
        
        # Process batch if we've reached max size
        if len(self.current_batch) >= self.max_batch_size:
            self._process_batch()
    
    def _should_process_batch(self) -> bool:
        """Check if we should process the current batch based on timeout."""
        if not self.current_batch:
            return False
        
        elapsed = time.time() - self.batch_start_time
        return elapsed >= self.batch_timeout_ms
    
    def _process_batch(self):
        """Process the current batch of requests."""
        if not self.current_batch:
            return
        
        batch_size = len(self.current_batch)
        print(f"InferenceServer: Processing batch of {batch_size} requests")
        
        try:
            # Prepare input tensors
            fens = [req.fen for req in self.current_batch]
            x = np.array([tokenizer.tokenize(fen) for fen in fens])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            
            # Run inference with the correct device
            with torch.inference_mode(), autocast(self.device, dtype=torch.bfloat16):
                output = self.model(x)
            
            # Convert outputs to CPU and numpy for transmission
            result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.float().cpu().numpy()
                else:
                    result[key] = value
            
            # Send individual responses back to each requester
            for i, request in enumerate(self.current_batch):
                # Extract this request's output from the batch
                individual_output = {}
                for key, batch_value in result.items():
                    if isinstance(batch_value, np.ndarray) and len(batch_value.shape) > 0:
                        individual_output[key] = batch_value[i:i+1]  # Keep batch dimension
                    else:
                        individual_output[key] = batch_value
                
                response = InferenceResponse(request.request_id, individual_output)
                try:
                    self.response_queue.put(response, timeout=1.0)
                except queue.Full:
                    print(f"Warning: Response queue full for request {request.request_id}")
                    
        except Exception as e:
            print(f"InferenceServer batch processing error: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error responses
            for request in self.current_batch:
                error_response = InferenceResponse(request.request_id, {}, str(e))
                try:
                    self.response_queue.put(error_response, timeout=1.0)
                except queue.Full:
                    pass
        
        finally:
            # Clear the batch
            self.current_batch = []
    
    def shutdown(self):
        """Shutdown the inference server."""
        try:
            self.request_queue.put("SHUTDOWN", timeout=1.0)
        except queue.Full:
            pass
        self.join(timeout=5.0)
        if self.is_alive():
            self.terminate() 