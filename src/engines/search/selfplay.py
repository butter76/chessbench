import time
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from src.engines.search.alphabeta_worker import AlphaBetaWorker
from src.engines.search.utils import SearchManager
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
import numpy as np

from searchless_chess.src.models.transformer import ChessTransformer, TransformerConfig



if __name__ == "__main__":

    checkpoint_path = "checkpoints/self-play/T0/gen0.pt"
    checkpoint = torch.load(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = checkpoint['model_config']

    model = ChessTransformer(model_config).to(device)

    if checkpoint['compiled']:
        model = torch.compile(model)
    model.load_state_dict(checkpoint['model'])

    model.eval()


    def evaluate(x: np.ndarray):
        """Evaluate a position using a neural network."""
        N = x.shape[0]
        x = torch.from_numpy(x).to(device)

        with torch.inference_mode(), autocast(device, dtype=torch.bfloat16):
            output = model(x)
        values = output['value'][:, 0] * 2.0 - 1.0
        is_legal = output['legal'] > 0 # Rather than manually checking if the move is legal, just use the model's prediction of legality
        policies = output['policy']
        policies[~is_legal] = float('-inf')
        policies = torch.nn.functional.softmax(policies.view(N, -1), dim=-1).view(N, is_legal.shape[1], -1)
        return values.cpu().numpy(), policies.cpu().numpy()

    # Create initial board
    board = chess.Board()

    # Create search manager with our  model
    search_manager = SearchManager(
        model_predict_fn=evaluate,
        max_batch_size=48,
        timeout=0.01,
        game_log_file="../data/self-play/gen0-selfplay.bag"
    )

    for i in range(96):
        alpha_beta_worker = AlphaBetaWorker(
            initial_board=board,
            evaluation_queue=search_manager.evaluation_queue,
            max_depth=3
        )
        search_manager.add_worker(alpha_beta_worker)

    time.sleep(60 * 15)

    search_manager.stop()

    
    



        


    