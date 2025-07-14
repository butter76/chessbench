import { TreeNode } from '../types/SearchLog';

export interface LastMove {
  from: string;
  to: string;
}

/**
 * Parses a UCI move string (e.g., "e2e4") into from and to squares
 */
export function parseUCIMove(moveString: string): LastMove | null {
  if (!moveString || moveString.length < 4) {
    return null;
  }

  const from = moveString.substring(0, 2);
  const to = moveString.substring(2, 4);

  // Basic validation for algebraic notation
  if (!/^[a-h][1-8]$/.test(from) || !/^[a-h][1-8]$/.test(to)) {
    return null;
  }

  return { from, to };
}

/**
 * Parses FEN notation to determine whose turn it is to move
 * @param fen FEN string like "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
 * @returns 'w' for white, 'b' for black, or null if invalid
 */
export function getActivePlayerFromFEN(fen: string): 'w' | 'b' | null {
  if (!fen) {
    return null;
  }

  const parts = fen.split(' ');
  if (parts.length < 2) {
    return null;
  }

  const activeColor = parts[1];
  if (activeColor === 'w' || activeColor === 'b') {
    return activeColor;
  }

  return null;
}

/**
 * Gets the border color for a chess position based on whose turn it is
 * @param fen FEN string
 * @returns CSS color string for the border
 */
export function getBorderColorForPosition(fen: string): string {
  const activePlayer = getActivePlayerFromFEN(fen);
  
  switch (activePlayer) {
    case 'w':
      return '#ffffff'; // White border for white to move
    case 'b':
      return '#000000'; // Black border for black to move
    default:
      return '#adb5bd'; // Default gray border if unable to determine
  }
}

/**
 * Determines the last move that led to a specific node by looking at its parent's potential children
 */
export function getLastMoveForNode(node: TreeNode, nodes: Map<string, TreeNode>): LastMove | null {
  if (!node.parentId) {
    // Root node has no last move
    return null;
  }

  const parentNode = nodes.get(node.parentId);
  if (!parentNode) {
    return null;
  }

  // Use the parent move information directly from the node
  if (node.parentMove) {
    return parseUCIMove(node.parentMove);
  }

  return null;
}

/**
 * Creates square styles for highlighting the last move
 */
export function createLastMoveStyles(lastMove: LastMove | null): { [square: string]: React.CSSProperties } {
  if (!lastMove) {
    return {};
  }

  const highlightStyle: React.CSSProperties = {
    backgroundColor: 'rgba(255, 255, 0, 0.4)',
    boxShadow: 'inset 0 0 0 2px rgba(255, 255, 0, 0.8)',
  };

  return {
    [lastMove.from]: highlightStyle,
    [lastMove.to]: highlightStyle,
  };
} 