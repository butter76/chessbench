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

  // Find the move that led to this node by matching the child node
  // Children are stored in the order they were expanded, corresponding to potential_children order
  const childIndex = parentNode.children.findIndex(child => child.id === node.id);
  
  if (childIndex >= 0 && childIndex < parentNode.potentialChildren.length) {
    const potentialChild = parentNode.potentialChildren[childIndex];
    return parseUCIMove(potentialChild.move);
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