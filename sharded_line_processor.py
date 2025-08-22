import argparse
import multiprocessing as mp
import os
import signal
import sys
import chess
from typing import List, Optional, Tuple
import math

from searchless_chess.src.bagz import BagWriter
from searchless_chess.src.constants import LC0DataRecord, encode_lc0_data


def process_line(line_text: str, row_index: int, shard_index: int) -> None:
    """
    Placeholder for actual processing logic per line.
    Implement your real processing here.
    """
    # Example no-op to avoid optimizer removing loop
    entries = line_text.split("|")
    fen = entries[0]
    move, cp = entries[1].split("=")
    cp = int(cp)

    board = chess.Board(fen)
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    if len(board.piece_map()) <= 7:
        return None
    if not board.is_legal(chess.Move.from_uci(move)):
        return None

    policy = [(m.uci(), 1.0 if m.uci() == move else 0.0) for m in board.legal_moves]
    assert sum(p for _, p in policy) == 1.0, f'{policy} is not a valid policy'

    fen = board.fen()

    if (cp > 10000):
        eval = 1.0
    elif (cp < -10000):
        eval = -1.0
    else:
        eval = math.atan(cp / 90.0) / 1.563754
    d = 1 - abs(eval)

    return encode_lc0_data(LC0DataRecord(
        fen=fen,
        policy=policy,
        result=1.0 if cp > 100 else (-1.0 if cp < -100 else 0.0),
        root_q=eval,
        root_d=d,
        played_q=eval,
        played_d=d,
        plies_left=0,
        move="CDB",
    ))


    


def worker_consume(shard_index: int, input_queue: mp.Queue) -> None:
    """Continuously consume (row_index, line_text) tuples from the queue and process them."""
    try:
        output_bag = f"data/chessdb3_{shard_index:02d}17.bag"
        with BagWriter(output_bag) as writer:
            processed_count = 0
            while True:
                item: Optional[Tuple[int, str]] = input_queue.get()
                if item is None:
                    break
                row_index, line_text = item
                output = process_line(line_text=line_text, row_index=row_index, shard_index=shard_index)
                if output is not None:
                    writer.write(output)
                processed_count += 1
    except KeyboardInterrupt:
        # Allow graceful shutdown on Ctrl+C
        pass


def enqueue_lines(file_path: str, num_shards: int, queues: List[mp.Queue], start_index: int, strip_newline: bool) -> None:
    """
    Read the file sequentially and place each line into the queue for (row_index % num_shards).
    """
    buffering_bytes = 1024 * 1024  # 1MB buffer for efficient IO
    # Using UTF-8 default; adjust if your file is a different encoding
    with open(file_path, mode="r", encoding="utf-8", buffering=buffering_bytes) as infile:
        for row_index, line_text in enumerate(infile, start=start_index):
            if strip_newline:
                # Avoid carrying trailing newlines to workers
                line_text = line_text.rstrip("\n")
            shard_index = row_index % num_shards
            queues[shard_index].put((row_index, line_text))


def install_sigint_handler() -> None:
    """Ensure SIGINT (Ctrl+C) interrupts the main process cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shard processing of a large text file by line index modulo.")
    parser.add_argument(
        "--file",
        dest="file_path",
        default="/home/ubuntu/searchless_chess/data/output2.txt",
        help="Path to the input file (default: /home/ubuntu/searchless_chess/data/output2.txt)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=24,
        help="Number of shards/processes (default: 24)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10000,
        help="Max items buffered per shard queue before backpressure (default: 10000)",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Treat the first line as index 1 instead of 0 (default: 0-based)",
    )
    parser.add_argument(
        "--keep-newline",
        action="store_true",
        help="Keep trailing newline characters in line_text passed to workers (default: strip)",
    )

    args = parser.parse_args(argv)

    if args.num_shards <= 0:
        print("--num-shards must be positive", file=sys.stderr)
        return 2

    file_path: str = os.path.abspath(args.file_path)
    if not os.path.isfile(file_path):
        print(f"Input file not found: {file_path}", file=sys.stderr)
        return 2

    start_index: int = 1 if args.one_based else 0
    strip_newline: bool = not args.keep_newline

    # Ensure clean SIGINT handling in the main process
    install_sigint_handler()

    # Prefer fork on Linux for efficiency; ignore on platforms that do not support it
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        # Already set; continue
        pass

    # Create shard queues and worker processes
    queues: List[mp.Queue] = [mp.Queue(maxsize=args.queue_size) for _ in range(args.num_shards)]
    workers: List[mp.Process] = []

    for shard_index in range(args.num_shards):
        p = mp.Process(target=worker_consume, args=(shard_index, queues[shard_index]), daemon=False)
        p.start()
        workers.append(p)

    try:
        enqueue_lines(
            file_path=file_path,
            num_shards=args.num_shards,
            queues=queues,
            start_index=start_index,
            strip_newline=strip_newline,
        )
    except KeyboardInterrupt:
        # Allow user to interrupt; proceed to shutdown workers
        pass
    finally:
        # Signal completion to all shards with sentinel None
        for q in queues:
            q.put(None)

        # Wait for workers to complete processing
        for p in workers:
            try:
                p.join()
            except KeyboardInterrupt:
                # If interrupted again, attempt a quick shutdown
                p.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


