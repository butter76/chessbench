import React, { useState, useCallback, useMemo } from 'react';
import { 
  ReactFlow, 
  Background, 
  Controls, 
  MiniMap, 
  useNodesState, 
  useEdgesState,
  Node,
  ReactFlowProvider,
  Panel
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { TreeNode } from '../types/SearchLog';
import { parseSearchLogs, formatNodeDetails } from '../utils/logParser';
import { convertTreeToReactFlow, autoLayoutNodes, ReactFlowNodeData } from '../utils/reactFlowUtils';
import ChessSearchNode from './ChessSearchNode';
import ChessSearchEdge from './ChessSearchEdge';

interface SearchTreeViewerProps {
  logText: string;
}

const nodeTypes = {
  chessSearchNode: ChessSearchNode,
};

const edgeTypes = {
  chessSearchEdge: ChessSearchEdge,
};

const SearchTreeViewer: React.FC<SearchTreeViewerProps> = ({ logText }) => {
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);

  const treeData = useMemo(() => {
    if (!logText.trim()) return null;
    return parseSearchLogs(logText);
  }, [logText]);

  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!treeData) return { nodes: [], edges: [] };
    return convertTreeToReactFlow(treeData);
  }, [treeData]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    const nodeData = node.data as ReactFlowNodeData;
    setSelectedNode(nodeData.node);
  }, []);

  const handleAutoLayout = useCallback(() => {
    if (nodes.length === 0) return;
    
    const layoutedNodes = autoLayoutNodes(nodes, edges);
    setNodes(layoutedNodes);
  }, [nodes, edges, setNodes]);

  const handleClearSelection = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // Component to render formatted node details
  const NodeDetailsPanel: React.FC<{ node: TreeNode }> = ({ node }) => {
    const [copySuccess, setCopySuccess] = useState(false);

    const handleCopyFEN = async () => {
      try {
        await navigator.clipboard.writeText(node.fen);
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (err) {
        console.error('Failed to copy FEN:', err);
      }
    };

    return (
      <div>
        {/* Basic Node Information */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 1fr', 
            gap: '10px 20px',
            fontSize: '13px'
          }}>
            <div><strong>Node ID:</strong> <span style={{ fontFamily: 'monospace', color: '#666' }}>{node.id}</span></div>
            <div><strong>Depth:</strong> {node.depth}</div>
            <div><strong>Value:</strong> <span style={{ color: node.value > 0 ? '#28a745' : node.value < 0 ? '#dc3545' : '#6c757d' }}>{node.value.toFixed(4)}</span></div>
            <div><strong>Uncertainty:</strong> {node.U.toFixed(4)}</div>
            <div><strong>Expected Value:</strong> {node.expval.toFixed(4)}</div>
            <div><strong>Expected Opp Value:</strong> {node.expoppval.toFixed(4)}</div>
            <div><strong>Terminal:</strong> {node.isTerminal ? 'Yes' : 'No'}</div>
            <div><strong>Children:</strong> {node.children.length} / {node.potentialChildren.length}</div>
          </div>
        </div>

        {/* FEN Position */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '5px' }}><strong>FEN Position:</strong></div>
          <div style={{ 
            fontFamily: 'monospace', 
            fontSize: '12px', 
            backgroundColor: '#f8f9fa', 
            padding: '8px', 
            borderRadius: '4px',
            wordBreak: 'break-all',
            border: '1px solid #e9ecef',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: '8px'
          }}>
            <span style={{ flex: 1 }}>{node.fen}</span>
            <button
              onClick={handleCopyFEN}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '0px',
                display: 'flex',
                alignItems: 'center',
                color: copySuccess ? '#28a745' : '#6c757d',
                fontSize: '12px',
                transition: 'color 0.2s',
                lineHeight: '1'
              }}
              title={copySuccess ? 'Copied!' : 'Copy FEN to clipboard'}
            >
              {copySuccess ? '✓' : '📋'}
            </button>
          </div>
        </div>

        {/* Potential Moves Table */}
        {node.potentialChildren.length > 0 && (
          <div>
            <div style={{ marginBottom: '10px' }}>
              <strong>Potential Moves ({node.potentialChildren.length}):</strong>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '12px',
                border: '1px solid #dee2e6'
              }}>
                <thead>
                  <tr style={{ backgroundColor: '#f8f9fa' }}>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'left', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>#</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'left', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>Move</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'center', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>UCI</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'right', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>Prob</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'right', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>U</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'right', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>Q</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'right', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>D</th>
                    <th style={{ 
                      padding: '8px 6px', 
                      textAlign: 'center', 
                      borderBottom: '1px solid #dee2e6',
                      fontWeight: 'bold'
                    }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {node.potentialChildren.map((child, index) => {
                    const isExpanded = index < node.children.length;
                    return (
                      <tr key={index} style={{ 
                        backgroundColor: isExpanded ? '#e8f5e8' : '#ffffff',
                        borderBottom: '1px solid #dee2e6'
                      }}>
                        <td style={{ 
                          padding: '6px', 
                          fontFamily: 'monospace',
                          color: '#495057'
                        }}>{index + 1}</td>
                        <td style={{ 
                          padding: '6px', 
                          fontFamily: 'monospace',
                          fontWeight: 'bold',
                          color: '#212529'
                        }}>{child.move_san}</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'center',
                          fontFamily: 'monospace',
                          color: '#6c757d',
                          fontSize: '11px'
                        }}>{child.move}</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          color: '#007bff'
                        }}>{(child.probability * 100).toFixed(1)}%</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          color: '#6c757d'
                        }}>{child.U.toFixed(3)}</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          color: child.Q > 0 ? '#28a745' : child.Q < 0 ? '#dc3545' : '#6c757d'
                        }}>{child.Q.toFixed(3)}</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          color: '#6c757d'
                        }}>{child.D.toFixed(3)}</td>
                        <td style={{ 
                          padding: '6px', 
                          textAlign: 'center'
                        }}>
                          {isExpanded ? (
                            <span style={{ 
                              color: '#28a745',
                              fontWeight: 'bold',
                              fontSize: '11px'
                            }}>EXPANDED</span>
                          ) : (
                            <span style={{ 
                              color: '#6c757d',
                              fontSize: '11px'
                            }}>NOT EXPANDED</span>
                          )}
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
    );
  };

  if (!treeData || !treeData.root) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <p>No search tree data available. Please load search logs.</p>
      </div>
    );
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.1}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        >
          <Background />
          <Controls />
          <MiniMap 
            nodeColor={(node) => {
              const nodeData = node.data as ReactFlowNodeData;
              if (nodeData?.node?.isTerminal) return '#ff6b6b';
              return '#4ecdc4';
            }}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
          
          <Panel position="top-left">
            <div style={{
              backgroundColor: 'white',
              padding: '10px',
              borderRadius: '8px',
              border: '1px solid #ddd',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              display: 'flex',
              gap: '10px',
              alignItems: 'center',
            }}>
              <button
                onClick={handleAutoLayout}
                style={{
                  padding: '5px 10px',
                  fontSize: '12px',
                  backgroundColor: '#4ecdc4',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Auto Layout
              </button>
              
              <button
                onClick={handleClearSelection}
                style={{
                  padding: '5px 10px',
                  fontSize: '12px',
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Clear Selection
              </button>
              
              <div style={{
                fontSize: '12px',
                color: '#666',
                fontFamily: 'monospace',
              }}>
                Nodes: {nodes.length} | Max Depth: {treeData.maxDepth}
              </div>
            </div>
          </Panel>
        </ReactFlow>
      </ReactFlowProvider>

      {/* Node details panel */}
      {selectedNode && (
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          width: '650px',
          maxHeight: 'calc(100vh - 40px)',
          backgroundColor: '#ffffff',
          border: '1px solid #dee2e6',
          borderRadius: '8px',
          padding: '20px',
          overflowY: 'auto',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
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
                color: '#495057'
              }}
            >
              ×
            </button>
          </div>
          
          <NodeDetailsPanel node={selectedNode} />
        </div>
      )}
    </div>
  );
};

export default SearchTreeViewer; 