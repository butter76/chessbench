# Rust Chess Engine

This directory contains a Rust implementation of chess logic with Python bindings.

## Project Structure

- `chess-bindings/`: PyO3 bindings to expose the chess engine to Python

## Building the Project

The project uses [maturin](https://github.com/PyO3/maturin) to build the Python bindings. To build the project:

```bash
# Install maturin if you haven't already
pip install maturin

# Development build (faster, but not optimized)
maturin develop

# Production build
maturin build --release
```

## Using in Python

After building, you can import and use the chess engine in Python:

```python
from rust_chess import ChessEngine

# Create a new chess engine
engine = ChessEngine()

# Get legal moves
moves = engine.get_legal_moves()
print(moves)

# Make a move
engine.make_move("e2e4")
```

See `src/rust_chess/example.py` for a complete example.

## Development

### Testing the Rust Code

```bash
cd rust
cargo test
```

### Adding New Features

1. Expose it through the PyO3 bindings in `chess-bindings`
2. Update the Python wrapper in `src/rust_chess/__init__.py` 