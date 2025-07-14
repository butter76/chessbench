import { Node, Edge } from '@xyflow/react';
import { TreeNode, TreeStructure } from '../types/SearchLog';

export interface ReactFlowNodeData extends Record<string, unknown> {
  node: TreeNode;
  nodes: Map<string, TreeNode>;
}

export interface ReactFlowEdgeData extends Record<string, unknown> {
  probability: number;
  move: string;
  move_san: string;
}

export function convertTreeToReactFlow(treeData: TreeStructure): {
  nodes: Node<ReactFlowNodeData>[];
  edges: Edge<ReactFlowEdgeData>[];
} {
  const nodes: Node<ReactFlowNodeData>[] = [];
  const edges: Edge<ReactFlowEdgeData>[] = [];

  if (!treeData.root) {
    return { nodes, edges };
  }

  // Calculate positions using a simple hierarchical layout
  const nodePositions = new Map<string, { x: number; y: number }>();
  const levelNodes: TreeNode[][] = [];
  
  // Group nodes by depth level
  const visitNode = (node: TreeNode, level: number) => {
    if (!levelNodes[level]) {
      levelNodes[level] = [];
    }
    levelNodes[level].push(node);
    
    node.children.forEach(child => {
      visitNode(child, level + 1);
    });
  };

  visitNode(treeData.root, 0);

  // Position nodes
  const NODE_WIDTH = 160;
  const LEVEL_HEIGHT = 250;
  
  levelNodes.forEach((levelNodeList, level) => {
    const levelWidth = levelNodeList.length * NODE_WIDTH;
    const startX = -levelWidth / 2;
    
    levelNodeList.forEach((node, index) => {
      const x = startX + (index * NODE_WIDTH) + (NODE_WIDTH / 2);
      const y = level * LEVEL_HEIGHT;
      nodePositions.set(node.id, { x, y });
    });
  });

  // Create React Flow nodes
  const traverseForNodes = (node: TreeNode) => {
    const position = nodePositions.get(node.id) || { x: 0, y: 0 };
    
    nodes.push({
      id: node.id,
      type: 'chessSearchNode',
      position,
      data: {
        node,
        nodes: treeData.nodes,
      },
    });

    // Create edges for children
    node.children.forEach((child) => {
      // Find the corresponding potential child for this connection by matching parent move
      let probability = 0;
      let move = '';
      let move_san = '';
      
      if (child.parentMove) {
        const potentialChild = node.potentialChildren.find(pc => pc.move === child.parentMove);
        if (potentialChild) {
          probability = potentialChild.probability;
          move = potentialChild.move;
          move_san = potentialChild.move_san;
        }
      }

      edges.push({
        id: `${node.id}-${child.id}`,
        source: node.id,
        target: child.id,
        type: 'chessSearchEdge',
        data: {
          probability,
          move,
          move_san,
        },
      });
    });

    // Recursively process children
    node.children.forEach(child => {
      traverseForNodes(child);
    });
  };

  traverseForNodes(treeData.root);

  return { nodes, edges };
}

// Auto-layout function using dagre for better positioning
export function autoLayoutNodes(nodes: Node<ReactFlowNodeData>[], edges: Edge<ReactFlowEdgeData>[]): Node<ReactFlowNodeData>[] {
  // For now, we'll use a simple hierarchical layout
  // In the future, we could integrate with dagre for more sophisticated layouts
  
  // Group nodes by their tree level (we can infer this from the node structure)
  const nodesByLevel = new Map<number, Node<ReactFlowNodeData>[]>();
  
  nodes.forEach(node => {
    const treeNode = (node.data as ReactFlowNodeData).node;
    const level = treeNode.depth;
    
    if (!nodesByLevel.has(level)) {
      nodesByLevel.set(level, []);
    }
    nodesByLevel.get(level)!.push(node);
  });

  // Re-position nodes to avoid overlaps
  const NODE_WIDTH = 160;
  const LEVEL_HEIGHT = 250;
  const MIN_NODE_SPACING = 20;

  const updatedNodes = nodes.map(node => {
    const treeNode = (node.data as ReactFlowNodeData).node;
    const level = treeNode.depth;
    const levelNodes = nodesByLevel.get(level) || [];
    const nodeIndex = levelNodes.findIndex(n => n.id === node.id);
    
    const totalWidth = levelNodes.length * (NODE_WIDTH + MIN_NODE_SPACING);
    const startX = -totalWidth / 2;
    const x = startX + (nodeIndex * (NODE_WIDTH + MIN_NODE_SPACING)) + (NODE_WIDTH / 2);
    const y = level * LEVEL_HEIGHT;

    return {
      ...node,
      position: { x, y },
    };
  });

  return updatedNodes;
} 