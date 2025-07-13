import { SearchLogEntry, TreeNode, TreeStructure, PotentialChild } from '../types/SearchLog';

export function parseSearchLogs(logText: string): TreeStructure {
  const lines = logText.trim().split('\n');
  const nodes = new Map<string, TreeNode>();
  let root: TreeNode | null = null;
  let maxDepth = 0;

  // Parse each JSON line
  const logEntries: SearchLogEntry[] = lines
    .filter(line => line.trim().length > 0)
    .map(line => {
      try {
        return JSON.parse(line) as SearchLogEntry;
      } catch (error) {
        console.warn('Failed to parse log line:', line, error);
        return null;
      }
    })
    .filter(entry => entry !== null) as SearchLogEntry[];

  // Sort by timestamp to ensure proper order
  logEntries.sort((a, b) => a.timestamp - b.timestamp);

  // Build tree structure
  for (const entry of logEntries) {
    const node: TreeNode = {
      id: entry.node_id,
      parentId: entry.parent_id,
      fen: entry.fen,
      value: entry.value,
      U: entry.U,
      expval: entry.expval,
      expoppval: entry.expoppval,
      isTerminal: entry.is_terminal,
      potentialChildren: entry.potential_children,
      children: [],
      timestamp: entry.timestamp,
      depth: 0, // Will be calculated later
    };

    nodes.set(entry.node_id, node);

    // Set root if this is the first node or has no parent
    if (!root && (entry.parent_id === null || entry.parent_id === undefined)) {
      root = node;
    }
  }

  // Build parent-child relationships and calculate depths
  for (const node of nodes.values()) {
    if (node.parentId) {
      const parent = nodes.get(node.parentId);
      if (parent) {
        parent.children.push(node);
        node.depth = parent.depth + 1;
        maxDepth = Math.max(maxDepth, node.depth);
      }
    }
  }

  // Note: Children are already added in the order they appear in the logs,
  // which should correspond to the order of expansion (highest probability first)

  return { nodes, root, maxDepth };
}

export function getNodePath(node: TreeNode, nodes: Map<string, TreeNode>): TreeNode[] {
  const path: TreeNode[] = [];
  let current: TreeNode | undefined = node;
  
  while (current) {
    path.unshift(current);
    current = current.parentId ? nodes.get(current.parentId) : undefined;
  }
  
  return path;
}

export function formatNodeDetails(node: TreeNode): string {
  const details = [
    `Node ID: ${node.id}`,
    `FEN: ${node.fen}`,
    `Value: ${node.value.toFixed(4)}`,
    `U (Uncertainty): ${node.U.toFixed(4)}`,
    `Expected Value: ${node.expval.toFixed(4)}`,
    `Expected Opponent Value: ${node.expoppval.toFixed(4)}`,
    `Terminal: ${node.isTerminal ? 'Yes' : 'No'}`,
    `Depth: ${node.depth}`,
    `Children: ${node.children.length}`,
    `Potential Children: ${node.potentialChildren.length}`,
  ];

  if (node.potentialChildren.length > 0) {
    details.push('');
    details.push('Potential Moves:');
    node.potentialChildren.forEach((child, index) => {
      // Simple heuristic: assume the top N moves (by probability) are expanded 
      // if there are N children, since moves are typically expanded in probability order
      const isExpanded = index < node.children.length;
      
      details.push(
        `  ${index + 1}. ${child.move_san} (${child.move}) - ` +
        `P: ${child.probability.toFixed(4)}, ` +
        `U: ${child.U.toFixed(4)}, ` +
        `Q: ${child.Q.toFixed(4)}, ` +
        `D: ${child.D.toFixed(4)} ` +
        `${isExpanded ? '[EXPANDED]' : '[NOT EXPANDED]'}`
      );
    });
  }

  return details.join('\n');
} 