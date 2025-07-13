import React from 'react';
import Chessboard from 'chessboardjsx';

interface ChessBoardProps {
  fen: string;
  size?: number;
}

const ChessBoard: React.FC<ChessBoardProps> = ({ fen, size = 120 }) => {
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
      />
    </div>
  );
};

export default ChessBoard; 