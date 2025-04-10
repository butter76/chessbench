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
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'C',
    'Q',
    'K',
    'w',
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
  for char in raw_board:
    if char in _SPACES_CHARACTERS:
      board += '.' * int(char)
    else:
      board += char
  if en_passant != '-':
    en_sq = _parse_square(en_passant)
    assert board[en_sq] == '.'
    board = board[:en_sq] + 'x' + board[en_sq + 1:]
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
  board = board + side

  assert board[-1] in ['w', 'b']

  indices = list()

  for char in board:
    indices.append(_CHARACTERS_INDEX[char])

  # if castling == '-':
  #   indices.extend(4 * [_CHARACTERS_INDEX['.']])
  # else:
  #   for char in castling:
  #     indices.append(_CHARACTERS_INDEX[char])
  #   # Padding castling to have exactly 4 characters.
  #   if len(castling) < 4:
  #     indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  # Three digits for halfmoves (since last capture) is enough since the game
  # ends at 50.
  halfmoves_last += '.' * (3 - len(halfmoves_last))
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  assert len(indices) == SEQUENCE_LENGTH

  return np.asarray(indices, dtype=np.uint8)
