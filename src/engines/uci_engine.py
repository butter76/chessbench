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

"""Implements a generic UCI chess engine wrapper."""

from typing import Sequence

import chess
import chess.engine

from searchless_chess.src.engines import engine


class UCIEngine(engine.Engine):
	"""A generic UCI chess engine wrapper."""

	def __init__(
			self,
			engine_path: str,
			limit: chess.engine.Limit,
			options: Sequence[str] | None = None,
			config: dict[str, str | int | bool | None] | None = None,
	) -> None:
		"""Initialize the UCI engine.
		
		Args:
			engine_path: Path to the UCI engine binary.
			limit: Time/depth/nodes limit for analysis.
			options: Optional command line options for engine startup.
			config: Optional UCI options to configure the engine.
		"""
		self._limit = limit
		command = [engine_path]
		if options:
			command.extend(options)
		
		self._raw_engine = chess.engine.SimpleEngine.popen_uci(command)
		
		if config:
			self._raw_engine.configure(config)

	def __del__(self) -> None:
		self._raw_engine.close()

	@property
	def limit(self) -> chess.engine.Limit:
		return self._limit

	def analyse(self, board: chess.Board) -> engine.AnalysisResult:
		"""Returns analysis results from the engine."""
		return self._raw_engine.analyse(board, limit=self._limit)

	def play(self, board: chess.Board) -> chess.Move:
		"""Returns the best move from the engine."""
		best_move = self._raw_engine.play(board, limit=self._limit).move
		if best_move is None:
			raise ValueError('No best move found, something went wrong.')
		return best_move


class AllMovesUCIEngine(UCIEngine):
	"""A version of UCI engine that evaluates all moves individually."""

	def analyse(self, board: chess.Board) -> engine.AnalysisResult:
		"""Returns analysis results evaluating each move individually."""
		scores = []
		sorted_legal_moves = engine.get_ordered_legal_moves(board)
		for move in sorted_legal_moves:
			results = self._raw_engine.analyse(
					board,
					limit=self._limit,
					root_moves=[move],
			)
			scores.append((move, results['score'].relative))
		return {'scores': scores}

	def play(self, board: chess.Board) -> chess.Move:
		"""Returns the best move after evaluating all moves."""
		scores = self.analyse(board)['scores']
		sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
		return sorted_scores[0][0]