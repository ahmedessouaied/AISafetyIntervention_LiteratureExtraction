from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from sentence_transformers import SentenceTransformer

from .core import Node, Edge, PaperSchema


class GraphNode(Node):
    """Extended Node class with embedding support."""
    embedding: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        # Handle embedding separately to avoid pydantic validation issues
        embedding = data.pop('embedding', None)
        super().__init__(**data)
        self.embedding = embedding


class GraphEdge(Edge):
    """Extended Edge class with embedding and concept metadata support."""
    embedding: Optional[np.ndarray] = None
    logical_chain_title: Optional[str] = None  # Equivalent to title in LogicalChain
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        # Handle embedding separately to avoid pydantic validation issues
        embedding = data.pop('embedding', None)
        logical_chain_title = data.pop('logical_chain_title', None)
        super().__init__(**data)
        self.embedding = embedding
        self.logical_chain_title = logical_chain_title


class LocalGraph(BaseModel):
    """Container for graph data with nodes and edges that have embeddings."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paper_id: str
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_model: Optional[SentenceTransformer] = None

    def __len__(self) -> int:
        """Return total number of nodes and edges."""
        return len(self.nodes) + len(self.edges)

    @classmethod
    def from_paper_schema(self, paper_schema: PaperSchema, json_path: Path) -> "LocalGraph":
        """Create a LocalGraph from a PaperSchema."""
        # Basic file-level checks
        names = [n.name for n in paper_schema.nodes]
        if len(names) != len(set(names)):
            dupes = sorted({x for x in names if names.count(x) > 1})
            raise ValueError(f"Duplicate node names in {json_path.name}: {dupes}")

        known = set(names)
        missing = [
            (e.source_node, e.target_node)
            for ch in paper_schema.logical_chains
            for e in ch.edges
            if e.source_node not in known or e.target_node not in known
        ]
        if missing:
            raise ValueError(f"Edges reference unknown nodes in {json_path.name}: {missing[:5]}...")

        # Convert to LocalGraph
        graph_nodes = [GraphNode(**node.model_dump()) for node in paper_schema.nodes]

        # Convert logical chains to edges with concept metadata
        graph_edges = []
        for logical_chain in paper_schema.logical_chains:
            for edge in logical_chain.edges:
                graph_edge = GraphEdge(**edge.model_dump(), logical_chain_title=logical_chain.title)
                graph_edges.append(graph_edge)
        local_graph = LocalGraph(nodes=graph_nodes, edges=graph_edges, paper_id=json_path.stem)
        return local_graph


    def _add_embeddings_to_nodes(self, node: GraphNode) -> None:
        """Add embeddings to all nodes in the local graph."""
        # Create text representation for embedding
        text_parts = []
        if node.name:
            text_parts.append(f"Name: {node.name}")
        if node.description:
            text_parts.append(f"Description: {node.description}")
        if node.aliases:
            text_parts.append(f"Aliases: {', '.join(node.aliases)}")
        if node.concept_category:
            text_parts.append(f"Category: {node.concept_category}")

        text = " | ".join(text_parts)
        node.embedding = self._get_embedding(text)

    def _add_embeddings_to_edges(self, edge: GraphEdge) -> None:
        """Add embeddings to all edges in the local graph."""
        # Create text representation for embedding
        text_parts = []
        if edge.type:
            text_parts.append(f"Type: {edge.type}")
        if edge.description:
            text_parts.append(f"Description: {edge.description}")
        if edge.logical_chain_title:
            text_parts.append(f"Concept: {edge.logical_chain_title}")
        if edge.source_node:
            text_parts.append(f"From: {edge.source_node}")
        if edge.target_node:
            text_parts.append(f"To: {edge.target_node}")

        text = " | ".join(text_parts)
        edge.embedding = self._get_embedding(text)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text using SentenceTransformers."""
        try:
            # Lazy load the model
            if self.embedding_model is None:
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Get embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            # Return zero vector as fallback (BGE-large-v1.5 has 1024 dimensions)
            return np.zeros(1024, dtype=np.float32)