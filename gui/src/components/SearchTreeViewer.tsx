import React, { useState, useCallback, useMemo } from 'react';
import { Group } from '@visx/group';
import { Tree, hierarchy } from '@visx/hierarchy';
import { LinkHorizontal } from '@visx/shape';
import { Zoom } from '@visx/zoom';
import { TreeNode, TreeStructure } from '../types/SearchLog';
import { parseSearchLogs, formatNodeDetails } from '../utils/logParser';

interface SearchTreeViewerProps {
  logText: string;
}

interface TreeNodeWithCoords extends TreeNode {
  x: number;
  y: number;
  children: TreeNodeWithCoords[];
}

const NODE_WIDTH = 120;
const NODE_HEIGHT = 40;
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

  const getNodeColor = useCallback((node: TreeNode) => {
    if (node.isTerminal) return '#ff6b6b';
    if (node.value > 0.5) return '#51cf66';
    if (node.value < -0.5) return '#ff8cc8';
    return '#74c0fc';
  }, []);

  const getNodeBorderColor = useCallback((node: TreeNode) => {
    return selectedNode?.id === node.id ? '#339af0' : '#adb5bd';
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
    <div style={{ display: 'flex', height: '100vh' }}>
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
                      {tree.links().map((link, i) => (
                        <LinkHorizontal
                          key={i}
                          data={link}
                          stroke="#dee2e6"
                          strokeWidth="2"
                          fill="none"
                        />
                      ))}
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
                            <rect
                              x={-NODE_WIDTH / 2}
                              y={-NODE_HEIGHT / 2}
                              width={NODE_WIDTH}
                              height={NODE_HEIGHT}
                              fill={getNodeColor(nodeData)}
                              stroke={getNodeBorderColor(nodeData)}
                              strokeWidth={isSelected ? 3 : 1}
                              rx={8}
                              style={{
                                filter: isSelected ? 'drop-shadow(0 0 6px rgba(51, 154, 240, 0.4))' : 'none'
                              }}
                            />
                            <text
                              dy="-0.5em"
                              fontSize={12}
                              fontFamily="monospace"
                              textAnchor="middle"
                              fill="#212529"
                              style={{ pointerEvents: 'none' }}
                            >
                              {nodeData.value.toFixed(3)}
                            </text>
                            <text
                              dy="0.8em"
                              fontSize={10}
                              fontFamily="monospace"
                              textAnchor="middle"
                              fill="#495057"
                              style={{ pointerEvents: 'none' }}
                            >
                              {nodeData.children.length}/{nodeData.potentialChildren.length}
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
  );
};

export default SearchTreeViewer; 