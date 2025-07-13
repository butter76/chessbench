export interface PotentialChild {
  move: string;
  move_san: string;
  probability: number;
  U: number;
  Q: number;
  D: number;
}

export interface SearchLogEntry {
  event: 'node_expansion';
  node_id: string;
  parent_id: string | null;
  fen: string;
  value: number;
  U: number;
  expval: number;
  expoppval: number;
  is_terminal: boolean;
  potential_children: PotentialChild[];
  num_potential_children: number;
  timestamp: number;
}

export interface TreeNode {
  id: string;
  parentId: string | null;
  fen: string;
  value: number;
  U: number;
  expval: number;
  expoppval: number;
  isTerminal: boolean;
  potentialChildren: PotentialChild[];
  children: TreeNode[];
  timestamp: number;
  depth: number;
}

export interface TreeStructure {
  nodes: Map<string, TreeNode>;
  root: TreeNode | null;
  maxDepth: number;
} 