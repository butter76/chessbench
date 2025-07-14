import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { 
  ReactFlow, 
  Background, 
  Controls, 
  MiniMap, 
  useNodesState, 
  useEdgesState,
  Node,
  ReactFlowProvider,
  Panel,
  XYPosition
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
  const [draggedSubtree, setDraggedSubtree] = useState<Set<string>>(new Set());
  const [dragStartPositions, setDragStartPositions] = useState<Map<string, XYPosition>>(new Map());

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

  // Helper function to find all descendant node IDs
  const findSubtreeNodeIds = useCallback((nodeId: string, edges: any[]): Set<string> => {
    const subtreeIds = new Set<string>();
    subtreeIds.add(nodeId);
    
    const findChildren = (currentId: string) => {
      const childEdges = edges.filter(edge => edge.source === currentId);
      childEdges.forEach(edge => {
        if (!subtreeIds.has(edge.target)) {
          subtreeIds.add(edge.target);
          findChildren(edge.target);
        }
      });
    };
    
    findChildren(nodeId);
    return subtreeIds;
  }, []);

  // Handle drag start - identify the subtree and store initial positions
  const handleNodeDragStart = useCallback((event: React.MouseEvent, node: Node) => {
    const subtreeIds = findSubtreeNodeIds(node.id, edges);
    setDraggedSubtree(subtreeIds);
    
    // Store initial positions for all nodes in the subtree
    const initialPositions = new Map<string, XYPosition>();
    nodes.forEach(n => {
      if (subtreeIds.has(n.id)) {
        initialPositions.set(n.id, { x: n.position.x, y: n.position.y });
      }
    });
    setDragStartPositions(initialPositions);
  }, [nodes, edges, findSubtreeNodeIds]);

  // Handle drag - move the entire subtree
  const handleNodeDrag = useCallback((event: React.MouseEvent, node: Node) => {
    if (draggedSubtree.size === 0) return;
    
    const draggedNodeStartPos = dragStartPositions.get(node.id);
    if (!draggedNodeStartPos) return;
    
    // Calculate the offset from the original position
    const deltaX = node.position.x - draggedNodeStartPos.x;
    const deltaY = node.position.y - draggedNodeStartPos.y;
    
    // Update positions of all nodes in the subtree
    setNodes(currentNodes => 
      currentNodes.map(n => {
        if (draggedSubtree.has(n.id) && n.id !== node.id) {
          const startPos = dragStartPositions.get(n.id);
          if (startPos) {
            return {
              ...n,
              position: {
                x: startPos.x + deltaX,
                y: startPos.y + deltaY
              }
            };
          }
        }
        return n;
      })
    );
  }, [draggedSubtree, dragStartPositions, setNodes]);

  // Handle drag stop - clear the dragged subtree state
  const handleNodeDragStop = useCallback((event: React.MouseEvent, node: Node) => {
    setDraggedSubtree(new Set());
    setDragStartPositions(new Map());
  }, []);

  // Auto layout when tree data is first loaded
  useEffect(() => {
    if (treeData && nodes.length > 0) {
      const layoutedNodes = autoLayoutNodes(nodes, edges);
      setNodes(layoutedNodes);
    }
  }, [treeData, nodes.length, edges, setNodes]);

  // Update visual selection when selectedNode changes
  useEffect(() => {
    if (selectedNode) {
      setNodes(currentNodes => 
        currentNodes.map(node => ({
          ...node,
          selected: (node.data as ReactFlowNodeData).node.id === selectedNode.id
        }))
      );
    } else {
      // Clear all selections
      setNodes(currentNodes => 
        currentNodes.map(node => ({
          ...node,
          selected: false
        }))
      );
    }
  }, [selectedNode, setNodes]);

  // Update node styles to show which nodes are being dragged
  useEffect(() => {
    if (draggedSubtree.size > 0) {
      setNodes(currentNodes => 
        currentNodes.map(node => ({
          ...node,
          style: {
            ...node.style,
            opacity: draggedSubtree.has(node.id) ? 0.8 : 1.0,
            outline: draggedSubtree.has(node.id) ? '2px solid #4ecdc4' : 'none'
          }
        }))
      );
    } else {
      // Clear drag styling
      setNodes(currentNodes => 
        currentNodes.map(node => ({
          ...node,
          style: {
            ...node.style,
            opacity: 1.0,
            outline: 'none'
          }
        }))
      );
    }
  }, [draggedSubtree, setNodes]);

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

  // Auto layout function for subtrees
  const autoLayoutSubtree = useCallback((rootNodeId: string) => {
    const subtreeNodeIds = findSubtreeNodeIds(rootNodeId, edges);
    const subtreeNodes = nodes.filter(node => subtreeNodeIds.has(node.id));
    
    if (subtreeNodes.length === 0) return;
    
    // Find the root node in the subtree
    const rootNode = subtreeNodes.find(node => node.id === rootNodeId);
    if (!rootNode) return;
    
    // Group subtree nodes by their depth relative to the subtree root
    const rootDepth = (rootNode.data as ReactFlowNodeData).node.depth;
    const nodesByLevel = new Map<number, Node<ReactFlowNodeData>[]>();
    
    subtreeNodes.forEach(node => {
      const treeNode = (node.data as ReactFlowNodeData).node;
      const relativeLevel = treeNode.depth - rootDepth;
      
      if (!nodesByLevel.has(relativeLevel)) {
        nodesByLevel.set(relativeLevel, []);
      }
      nodesByLevel.get(relativeLevel)!.push(node);
    });
    
    // Layout constants
    const NODE_WIDTH = 160;
    const LEVEL_HEIGHT = 250;
    const MIN_NODE_SPACING = 20;
    
    // Calculate new positions relative to the root node
    const rootPosition = rootNode.position;
    const updatedPositions = new Map<string, { x: number; y: number }>();
    
    nodesByLevel.forEach((levelNodes, level) => {
      const totalWidth = levelNodes.length * (NODE_WIDTH + MIN_NODE_SPACING);
      const startX = rootPosition.x - totalWidth / 2;
      
      levelNodes.forEach((node, index) => {
        const x = startX + (index * (NODE_WIDTH + MIN_NODE_SPACING)) + (NODE_WIDTH / 2);
        const y = rootPosition.y + (level * LEVEL_HEIGHT);
        
        updatedPositions.set(node.id, { x, y });
      });
    });
    
    // Update node positions
    setNodes(currentNodes => 
      currentNodes.map(node => {
        const newPosition = updatedPositions.get(node.id);
        if (newPosition) {
          return {
            ...node,
            position: newPosition
          };
        }
        return node;
      })
    );
  }, [nodes, edges, findSubtreeNodeIds, setNodes]);

  const handleSubtreeAutoLayout = useCallback(() => {
    if (selectedNode) {
      autoLayoutSubtree(selectedNode.id);
    }
  }, [selectedNode, autoLayoutSubtree]);

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
        {/* Subtree Auto Layout Button */}
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={handleSubtreeAutoLayout}
            style={{
              width: '100%',
              padding: '8px 12px',
              fontSize: '13px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#0056b3';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#007bff';
            }}
            title={`Auto-layout this subtree to organize its ${findSubtreeNodeIds(node.id, edges).size} nodes`}
          >
            🔄 Auto Layout Subtree ({findSubtreeNodeIds(node.id, edges).size} nodes)
          </button>
        </div>

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
                    
                    const handleRowClick = () => {
                      if (isExpanded) {
                        // The expanded children are the first node.children.length entries
                        // in the potentialChildren array, so we can use the index directly
                        const childNode = node.children[index];
                        setSelectedNode(childNode);
                      }
                    };
                    
                    return (
                      <tr 
                        key={index} 
                        onClick={handleRowClick}
                        style={{ 
                          backgroundColor: isExpanded ? '#e8f5e8' : '#ffffff',
                          borderBottom: '1px solid #dee2e6',
                          cursor: isExpanded ? 'pointer' : 'default',
                          transition: 'background-color 0.2s'
                        }}
                        onMouseEnter={(e) => {
                          if (isExpanded) {
                            e.currentTarget.style.backgroundColor = '#d4edda';
                          }
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = isExpanded ? '#e8f5e8' : '#ffffff';
                        }}
                      >
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
          onNodeDragStart={handleNodeDragStart}
          onNodeDrag={handleNodeDrag}
          onNodeDragStop={handleNodeDragStop}
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
                display: 'flex',
                flexDirection: 'column',
                gap: '2px'
              }}>
                <div>Nodes: {nodes.length} | Max Depth: {treeData.maxDepth}</div>
                {draggedSubtree.size > 0 && (
                  <div style={{ 
                    color: '#4ecdc4',
                    fontWeight: 'bold',
                    fontSize: '11px'
                  }}>
                    Dragging subtree: {draggedSubtree.size} nodes
                  </div>
                )}
                <div style={{ 
                  fontSize: '10px',
                  color: '#999',
                  fontStyle: 'italic'
                }}>
                  💡 Drag any node to move its entire subtree
                </div>
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