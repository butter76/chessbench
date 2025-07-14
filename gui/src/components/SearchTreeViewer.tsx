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
      <div style={{ flex: 1, position: 'relative' }}>
        {/* Tree visualization */}
        <div style={{ width: '100%', height: '100%' }}>
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

        {/* Globally positioned node details panel */}
        {selectedNode && (
          <div style={{
            position: 'fixed',
            top: '80px', // Account for the control panel height
            right: '20px',
            width: '600px',
            maxHeight: 'calc(100vh - 100px)', // Leave space for control panel and margins
            backgroundColor: '#ffffff',
            border: '1px solid #dee2e6',
            borderRadius: '8px',
            padding: '20px',
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: '14px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            zIndex: 1000
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '15px',
              borderBottom: '1px solid #dee2e6',
              paddingBottom: '10px'
            }}>
              <h3 style={{ margin: '0', color: '#495057' }}>
                Node Details
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '18px',
                  cursor: 'pointer',
                  color: '#6c757d',
                  padding: '0',
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f8f9fa';
                  e.currentTarget.style.color = '#dc3545';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.color = '#6c757d';
                }}
                title="Close details panel"
              >
                ×
              </button>
            </div>
            <div style={{ lineHeight: '1.5' }}>
              <div style={{ marginBottom: '15px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                  <div><strong>Node ID:</strong> {selectedNode.id}</div>
                  <div><strong>Depth:</strong> {selectedNode.depth}</div>
                  <div><strong>Value:</strong> {selectedNode.value.toFixed(4)}</div>
                  <div><strong>U (Uncertainty):</strong> {selectedNode.U.toFixed(4)}</div>
                  <div><strong>Expected Value:</strong> {selectedNode.expval.toFixed(4)}</div>
                  <div><strong>Expected Opp Value:</strong> {selectedNode.expoppval.toFixed(4)}</div>
                  <div><strong>Terminal:</strong> {selectedNode.isTerminal ? 'Yes' : 'No'}</div>
                  <div><strong>Children:</strong> {selectedNode.children.length}</div>
                </div>
                <div style={{ marginTop: '10px' }}>
                  <div><strong>FEN:</strong></div>
                  <div style={{ 
                    backgroundColor: '#f8f9fa', 
                    padding: '8px', 
                    borderRadius: '4px', 
                    fontSize: '12px',
                    wordBreak: 'break-all',
                    marginTop: '5px'
                  }}>
                    {selectedNode.fen}
                  </div>
                </div>
              </div>
              
              {selectedNode.potentialChildren.length > 0 && (
                <div>
                  <h4 style={{ margin: '0 0 10px 0', color: '#495057' }}>
                    Potential Moves ({selectedNode.potentialChildren.length})
                  </h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{
                      width: '100%',
                      borderCollapse: 'collapse',
                      fontSize: '12px'
                    }}>
                      <thead>
                        <tr style={{ backgroundColor: '#f8f9fa' }}>
                          <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #dee2e6' }}>#</th>
                          <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #dee2e6' }}>Move</th>
                          <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #dee2e6' }}>UCI</th>
                          <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>Prob</th>
                          <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>U</th>
                          <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>Q</th>
                          <th style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #dee2e6' }}>D</th>
                          <th style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid #dee2e6' }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedNode.potentialChildren.map((child, index) => {
                          const isExpanded = index < selectedNode.children.length;
                          return (
                            <tr key={index} style={{ 
                              backgroundColor: isExpanded ? '#e8f5e8' : 'transparent',
                              borderBottom: '1px solid #f1f3f4'
                            }}>
                              <td style={{ padding: '6px 8px' }}>{index + 1}</td>
                              <td style={{ padding: '6px 8px', fontWeight: 'bold' }}>{child.move_san}</td>
                              <td style={{ padding: '6px 8px', color: '#6c757d' }}>{child.move}</td>
                              <td style={{ padding: '6px 8px', textAlign: 'right' }}>{child.probability.toFixed(4)}</td>
                              <td style={{ padding: '6px 8px', textAlign: 'right' }}>{child.U.toFixed(4)}</td>
                              <td style={{ padding: '6px 8px', textAlign: 'right' }}>{child.Q.toFixed(4)}</td>
                              <td style={{ padding: '6px 8px', textAlign: 'right' }}>{child.D.toFixed(4)}</td>
                              <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                                <span style={{
                                  padding: '2px 6px',
                                  borderRadius: '3px',
                                  fontSize: '10px',
                                  backgroundColor: isExpanded ? '#d4edda' : '#f8d7da',
                                  color: isExpanded ? '#155724' : '#721c24'
                                }}>
                                  {isExpanded ? 'EXPANDED' : 'NOT EXPANDED'}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Legend - positioned when no node is selected */}
        {!selectedNode && (
          <div style={{
            position: 'fixed',
            top: '80px',
            right: '20px',
            width: '400px',
            backgroundColor: '#ffffff',
            border: '1px solid #dee2e6',
            borderRadius: '8px',
            padding: '20px',
            fontFamily: 'monospace',
            fontSize: '14px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            zIndex: 1000
          }}>
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
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchTreeViewer; 