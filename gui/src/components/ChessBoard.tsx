import React from 'react';
import Chessboard from 'chessboardjsx';
import { LastMove, createLastMoveStyles } from '../utils/moveUtils';

interface ChessBoardProps {
  fen: string;
  size?: number;
  lastMove?: LastMove | null;
}

const ChessBoard: React.FC<ChessBoardProps> = ({ fen, size = 120, lastMove }) => {
  const squareStyles = createLastMoveStyles(lastMove ?? null);

  return (
    <div style={{ width: size, height: size }}>
      <Chessboard
        position={fen}
        width={size}
        draggable={false}
        orientation="white"
        lightSquareStyle={{
          backgroundColor: '#f0d9b5',
        }}
        darkSquareStyle={{
          backgroundColor: '#b58863',
        }}
        squareStyles={squareStyles}
      />
    </div>
  );
};

export default ChessBoard; 