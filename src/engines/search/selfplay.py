import time
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from searchless_chess.src.engines.search.alphabeta_worker import AlphaBetaWorker
from searchless_chess.src.engines.search.utils import SearchManager
torch.set_default_dtype(torch.float32)
torch.set_printoptions(profile="full")
import numpy as np

from searchless_chess.src.models.transformer import ChessTransformer, TransformerConfig



if __name__ == "__main__":

    checkpoint_path = "../checkpoints/self-play/T0/gen0.pt"
    checkpoint = torch.load(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        policies = output['policy'].clone()
        policies[~is_legal] = float('-inf')
        policies = torch.nn.functional.softmax(policies.view(N, -1), dim=-1).view(N, is_legal.shape[1], -1)
        return values.float().cpu().numpy(), policies.float().cpu().numpy()

    # Create search manager with our  model
    search_manager = SearchManager(
        model_predict_fn=evaluate,
        max_batch_size=16,
        timeout=0.01,
        game_log_file="../data/self-play/gen0-selfplay.bag",
        opening_book="../data/opening_book.txt",
    )

    for i in range(32):
        alpha_beta_worker = AlphaBetaWorker(
            evaluation_queue=search_manager.evaluation_queue,
            game_logger=search_manager.game_logger,
            search_manager=search_manager,
            max_depth=3
        )
        search_manager.add_worker(alpha_beta_worker)

    search_manager.start()

    time.sleep(60 * 60 * 8)

    search_manager.stop()

    
    



        


    