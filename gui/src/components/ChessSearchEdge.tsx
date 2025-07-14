import React, { memo } from 'react';
import { EdgeProps, getSmoothStepPath } from '@xyflow/react';

interface ChessSearchEdgeData extends Record<string, unknown> {
  probability: number;
  move: string;
  move_san: string;
}

const ChessSearchEdge: React.FC<EdgeProps> = memo(({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}) => {
  const edgeData = data as ChessSearchEdgeData;
  const probability = edgeData?.probability || 0;
  const move_san = edgeData?.move_san || '';
  
  // Calculate stroke width based on probability (1-8 range)
  const strokeWidth = Math.max(1, Math.min(8, 1 + (probability * 7)));
  
  // Get the path for the edge
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      {/* Main edge path */}
      <path
        id={id}
        style={{
          ...style,
          strokeWidth,
          stroke: '#dee2e6',
          fill: 'none',
        }}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      
      {/* Label for probability and move */}
      {probability > 0 && (
        <g>
          {/* Background rectangle for label */}
          <rect
            x={labelX - 25}
            y={labelY - 12}
            width={50}
            height={24}
            fill="rgba(255, 255, 255, 0.9)"
            stroke="#dee2e6"
            strokeWidth={0.5}
            rx={4}
          />
          
          {/* Move notation */}
          <text
            x={labelX}
            y={labelY - 2}
            textAnchor="middle"
            fontSize={10}
            fontFamily="monospace"
            fill="#212529"
            className="react-flow__edge-text"
          >
            {move_san}
          </text>
          
          {/* Probability percentage */}
          <text
            x={labelX}
            y={labelY + 8}
            textAnchor="middle"
            fontSize={9}
            fontFamily="monospace"
            fill="#495057"
            className="react-flow__edge-text"
          >
            {(probability * 100).toFixed(1)}%
          </text>
        </g>
      )}
    </>
  );
});

ChessSearchEdge.displayName = 'ChessSearchEdge';

export default ChessSearchEdge; 