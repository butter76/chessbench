import React, { useState, useCallback, useMemo } from 'react';
import { Group } from '@visx/group';
import { Tree, hierarchy } from '@visx/hierarchy';
import { LinkHorizontal } from '@visx/shape';
import { Zoom } from '@visx/zoom';
import { TreeNode } from '../types/SearchLog';
import { parseSearchLogs, formatNodeDetails } from '../utils/logParser';
import { getLastMoveForNode, getBorderColorForPosition } from '../utils/moveUtils';
import ChessBoard from './ChessBoard';

interface SearchTreeViewerProps {
  logText: string;
}



const NODE_WIDTH = 140;
const NODE_HEIGHT = 160;
const MARGIN = { top: 20, left: 40, right: 40, bottom: 20 };

const SearchTreeViewer: React.FC<SearchTreeViewerProps> = ({ logText }) => {
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [treeWidth, setTreeWidth] = useState(1200);
  const [treeHeight, setTreeHeight] = useState(800);

  const treeData = useMemo(() => {
    if (!logText.trim()) return null;
    return parseSearchLogs(logText);
  }, [logText]);

  const handleNodeClick = useCallback((node: TreeNode) => {
    setSelectedNode(node);
  }, []);



  const getNodeBorderColor = useCallback((node: TreeNode) => {
    return getBorderColorForPosition(node.fen);
  }, []);

  const isNodeSelected = useCallback((node: TreeNode) => {
    return selectedNode?.id === node.id;
  }, [selectedNode]);

  if (!treeData || !treeData.root) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <p>No search tree data available. Please load search logs.</p>
      </div>
    );
  }

  const rootHierarchy = hierarchy(treeData.root, (d) => d.children);
  const innerWidth = treeWidth - MARGIN.left - MARGIN.right;
  const innerHeight = treeHeight - MARGIN.top - MARGIN.bottom;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Control Panel */}
      <div style={{
        backgroundColor: '#f8f9fa',
        borderBottom: '1px solid #dee2e6',
        padding: '10px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: '30px',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '14px', fontWeight: '500', minWidth: '50px' }}>
            Width:
          </label>
          <input
            type="range"
            min="800"
            max="10000"
            step="50"
            value={treeWidth}
            onChange={(e) => setTreeWidth(Number(e.target.value))}
            style={{ width: '150px' }}
          />
          <span style={{ fontSize: '14px', minWidth: '60px', fontFamily: 'monospace' }}>
            {treeWidth}px
          </span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '14px', fontWeight: '500', minWidth: '50px' }}>
            Height:
          </label>
          <input
            type="range"
            min="600"
            max="10000"
            step="50"
            value={treeHeight}
            onChange={(e) => setTreeHeight(Number(e.target.value))}
            style={{ width: '150px' }}
          />
          <span style={{ fontSize: '14px', minWidth: '60px', fontFamily: 'monospace' }}>
            {treeHeight}px
          </span>
        </div>
        
        <button
          onClick={() => {
            setTreeWidth(1200);
            setTreeHeight(800);
          }}
          style={{
            padding: '5px 10px',
            fontSize: '12px',
            backgroundColor: '#e9ecef',
            border: '1px solid #ced4da',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Reset
        </button>
      </div>
      
      {/* Main content area */}
      <div style={{ display: 'flex', flex: 1 }}>
        {/* Tree visualization */}
        <div style={{ flex: 1, position: 'relative' }}>
        <Zoom<SVGSVGElement>
          width={treeWidth}
          height={treeHeight}
          scaleXMin={0.1}
          scaleXMax={4}
          scaleYMin={0.1}
          scaleYMax={4}
          initialTransformMatrix={{
            scaleX: 1,
            scaleY: 1,
            translateX: MARGIN.left,
            translateY: MARGIN.top,
            skewX: 0,
            skewY: 0,
          }}
        >
          {(zoom) => (
            <svg
              width={treeWidth}
              height={treeHeight}
              style={{ cursor: zoom.isDragging ? 'grabbing' : 'grab' }}
              ref={zoom.containerRef}
            >
              <rect
                width={treeWidth}
                height={treeHeight}
                rx={14}
                fill="#f8f9fa"
                onTouchStart={zoom.dragStart}
                onTouchMove={zoom.dragMove}
                onTouchEnd={zoom.dragEnd}
                onMouseDown={zoom.dragStart}
                onMouseMove={zoom.dragMove}
                onMouseUp={zoom.dragEnd}
                onMouseLeave={() => {
                  if (zoom.isDragging) zoom.dragEnd();
                }}
              />
              
              <Group transform={zoom.toString()}>
                <Tree<TreeNode>
                  root={rootHierarchy}
                  size={[innerHeight, innerWidth]}
                  separation={(a, b) => (a.parent === b.parent ? 1 : 2) / a.depth}
                >
                  {(tree) => (
                    <Group top={MARGIN.top} left={MARGIN.left}>
                      {tree.links().map((link, i) => {
                        const parentNode = link.source.data;
                        const childNode = link.target.data;
                        
                        // Find the probability for this connection
                        let probability = 0;
                        
                        if (parentNode.potentialChildren.length > 0) {
                          // Find the index of this child in the parent's children array
                          const childIndex = parentNode.children.findIndex(child => child.id === childNode.id);
                          
                          // Get the corresponding potential child (they're ordered by probability)
                          if (childIndex >= 0 && childIndex < parentNode.potentialChildren.length) {
                            const potentialChild = parentNode.potentialChildren[childIndex];
                            probability = potentialChild.probability;
                          }
                        }
                        
                        // Calculate stroke width based on probability (1-8 range)
                        const strokeWidth = Math.max(1, Math.min(8, 1 + (probability * 7)));
                        
                        // Calculate midpoint for text placement
                        const midX = (link.source.y + link.target.y) / 2;
                        const midY = (link.source.x + link.target.x) / 2;
                        
                        return (
                          <Group key={i}>
                            <LinkHorizontal
                              data={link}
                              stroke="#dee2e6"
                              strokeWidth={strokeWidth}
                              fill="none"
                            />
                            {probability > 0 && (
                              <Group>
                                {/* Background for text */}
                                <rect
                                  x={midX - 20}
                                  y={midY - 8}
                                  width={40}
                                  height={16}
                                  fill="rgba(255, 255, 255, 0.9)"
                                  stroke="#dee2e6"
                                  strokeWidth={0.5}
                                  rx={4}
                                />
                                {/* Probability percentage text */}
                                <text
                                  x={midX}
                                  y={midY}
                                  dy="0.3em"
                                  fontSize={10}
                                  fontFamily="monospace"
                                  textAnchor="middle"
                                  fill="#495057"
                                  style={{ pointerEvents: 'none' }}
                                >
                                  {(probability * 100).toFixed(1)}%
                                </text>
                              </Group>
                            )}
                          </Group>
                        );
                      })}
                      {tree.descendants().map((node, key) => {
                        const nodeData = node.data;
                        const isSelected = selectedNode?.id === nodeData.id;
                        
                        return (
                          <Group
                            key={key}
                            top={node.x}
                            left={node.y}
                            onClick={() => handleNodeClick(nodeData)}
                            style={{ cursor: 'pointer' }}
                          >
                            {/* Background container with double border effect */}
                            {/* Outer black border for visibility */}
                            <rect
                              x={-NODE_WIDTH / 2 - 1}
                              y={-NODE_HEIGHT / 2 - 1}
                              width={NODE_WIDTH + 2}
                              height={NODE_HEIGHT + 2}
                              fill="none"
                              stroke="#000000"
                              strokeWidth={1}
                              rx={8}
                            />
                            {/* Main background with colored border */}
                            <rect
                              x={-NODE_WIDTH / 2}
                              y={-NODE_HEIGHT / 2}
                              width={NODE_WIDTH}
                              height={NODE_HEIGHT}
                              fill="white"
                              stroke={getNodeBorderColor(nodeData)}
                              strokeWidth={3}
                              rx={8}
                              style={{
                                filter: isNodeSelected(nodeData) ? 'drop-shadow(0 0 8px rgba(51, 154, 240, 0.6))' : 'none'
                              }}
                            />
                            
                            {/* ChessBoard */}
                            <foreignObject
                              x={-NODE_WIDTH / 2 + 10}
                              y={-NODE_HEIGHT / 2 + 10}
                              width={NODE_WIDTH - 20}
                              height={NODE_WIDTH - 20}
                            >
                              <ChessBoard 
                                fen={nodeData.fen} 
                                size={NODE_WIDTH - 20} 
                                lastMove={getLastMoveForNode(nodeData, treeData.nodes)}
                              />
                            </foreignObject>
                            
                            {/* Value information */}
                            <text
                              y={NODE_HEIGHT / 2 - 15}
                              fontSize={10}
                              fontFamily="monospace"
                              textAnchor="middle"
                              fill="#212529"
                              style={{ pointerEvents: 'none' }}
                            >
                              Q: {nodeData.value.toFixed(3)}
                            </text>
                            <text
                              y={NODE_HEIGHT / 2 - 4}
                              fontSize={10}
                              fontFamily="monospace"
                              textAnchor="middle"
                              fill="#495057"
                              style={{ pointerEvents: 'none' }}
                            >
                              U: {nodeData.U.toFixed(3)}
                            </text>
                          </Group>
                        );
                      })}
                    </Group>
                  )}
                </Tree>
              </Group>
            </svg>
          )}
        </Zoom>
        </div>

        {/* Node details panel */}
        <div style={{
          width: '400px',
          backgroundColor: '#ffffff',
          borderLeft: '1px solid #dee2e6',
          padding: '20px',
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '14px'
        }}>
          {selectedNode ? (
            <div>
              <h3 style={{ margin: '0 0 15px 0', color: '#495057' }}>
                Node Details
              </h3>
              <pre style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                margin: 0,
                lineHeight: '1.5'
              }}>
                {formatNodeDetails(selectedNode)}
              </pre>
            </div>
          ) : (
            <div style={{ color: '#6c757d', textAlign: 'center' }}>
              <p>Click on a node to view details</p>
              <div style={{ marginTop: '30px' }}>
                <h4>Legend:</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '20px', height: '20px', backgroundColor: '#51cf66', borderRadius: '4px' }}></div>
                    <span>Positive value (&gt; 0.5)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '20px', height: '20px', backgroundColor: '#ff8cc8', borderRadius: '4px' }}></div>
                    <span>Negative value (&lt; -0.5)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '20px', height: '20px', backgroundColor: '#74c0fc', borderRadius: '4px' }}></div>
                    <span>Neutral value</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '20px', height: '20px', backgroundColor: '#ff6b6b', borderRadius: '4px' }}></div>
                    <span>Terminal node</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchTreeViewer; 