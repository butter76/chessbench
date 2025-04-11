# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements tokenization of FEN strings."""

import numpy as np
from searchless_chess.src.utils import _parse_square


# pyfmt: disable
_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'p',
    'b',
    'n',
    'r',
    'c',
    'q',
    'k',
    'P',
    'B',
    'N',
    'R',
    'C',
    'Q',
    'K',
    'x',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 68


def tokenize(fen: str):
  """Returns an array of tokens from a fen string.

  We compute a tokenized representation of the board, from the FEN string.
  The final array of tokens is a mapping from this string to numbers, which
  are defined in the dictionary `_CHARACTERS_INDEX`.
  For the 'en passant' information, we convert the '-' (which means there is
  no en passant relevant square) to '..', to always have two characters, and
  a fixed length output.

  Args:
    fen: The board position in Forsyth-Edwards Notation.
  """
  # Extracting the relevant information from the FEN.
  raw_board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  raw_board = raw_board.replace('/', '')
  board = ''
  flip = side == 'b'
  for char in raw_board:
    if char in _SPACES_CHARACTERS:
      board += '.' * int(char)
    else:
      board += char
  for char in castling:
    if char == 'K':
      white_k_rook = _parse_square("h1")
      assert board[white_k_rook] == 'R'
      board = board[:white_k_rook] + 'C' + board[white_k_rook + 1:]
    elif char == 'Q':
      white_q_rook = _parse_square("a1")
      assert board[white_q_rook] == 'R'
      board = board[:white_q_rook] + 'C' + board[white_q_rook + 1:]
    elif char == 'k':
      black_k_rook = _parse_square("h8")
      assert board[black_k_rook] == 'r'
      board = board[:black_k_rook] + 'c' + board[black_k_rook + 1:]
    elif char == 'q':
      black_q_rook = _parse_square("a8")
      assert board[black_q_rook] == 'r'
      board = board[:black_q_rook] + 'c' + board[black_q_rook + 1:]
  if flip:
    board = board[56:64] + board[48:56] + board[40:48] + board[32:40] + board[24:32] + board[16:24] + board[8:16] + board[0:8]
    board = board.swapcase()
  if en_passant != '-':
    en_sq = _parse_square(en_passant, flip=flip)
    assert board[en_sq] == '.'
    board = board[:en_sq] + 'x' + board[en_sq + 1:]

  board += '.' * 4
  indices = list()

  for char in board:
    indices.append(_CHARACTERS_INDEX[char])

  assert len(board) == SEQUENCE_LENGTH

  planes = []
  planes.append(np.array([1 if char == 'p' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'P' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'b' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'B' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'n' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'N' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char in ['r', 'c'] else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char in ['R', 'C'] else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'c' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'C' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'q' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'Q' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'k' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'K' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char == 'x' else 0 for char in board], dtype=np.float32))
  planes.append(np.array([1 if char in ['.', 'x'] else 0 for char in board], dtype=np.float32))
  
  planes.append(np.array([int(halfmoves_last) / 100] * SEQUENCE_LENGTH, dtype=np.float32))

  planes = np.stack(planes).T
  assert len(indices) == SEQUENCE_LENGTH

  return planes, np.asarray(indices, dtype=np.int32)