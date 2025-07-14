import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import ChessBoard from './ChessBoard';
import { TreeNode } from '../types/SearchLog';
import { getLastMoveForNode, getBorderColorForPosition } from '../utils/moveUtils';

interface ChessSearchNodeData extends Record<string, unknown> {
  node: TreeNode;
  nodes: Map<string, TreeNode>;
}

const ChessSearchNode: React.FC<NodeProps> = memo(({ data, selected }) => {
  const { node, nodes } = data as ChessSearchNodeData;
  
  const borderColor = getBorderColorForPosition(node.fen);
  const lastMove = getLastMoveForNode(node, nodes);

  return (
    <div
      style={{
        padding: '8px',
        borderRadius: '8px',
        backgroundColor: 'white',
        border: `3px solid ${borderColor}`,
        boxShadow: selected ? '0 0 8px rgba(51, 154, 240, 0.6)' : '0 2px 4px rgba(0,0,0,0.1)',
        minWidth: '140px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
      }}
    >
      {/* Input handle for parent connections */}
      <Handle type="target" position={Position.Top} />
      
      {/* Chess board */}
      <ChessBoard 
        fen={node.fen} 
        size={120} 
        lastMove={lastMove}
      />
      
      {/* Node statistics */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '2px',
        fontSize: '10px',
        fontFamily: 'monospace',
        color: '#212529'
      }}>
        <div>Q: {node.value.toFixed(3)}</div>
        <div style={{ color: '#495057' }}>U: {node.U.toFixed(3)}</div>
      </div>
      
      {/* Output handle for child connections */}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

ChessSearchNode.displayName = 'ChessSearchNode';

export default ChessSearchNode; 