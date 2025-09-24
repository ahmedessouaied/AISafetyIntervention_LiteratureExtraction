from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


class Node(BaseModel):
    name: str = Field(..., description="concise natural-language description of node")
    type: Literal["concept", "intervention"]
    description: str = Field(
        ..., description="detailed technical description of node (1-2 sentences only)"
    )

    aliases: List[str] = Field(
        default_factory=list, description="2-3 alternative concise descriptions of node"
    )
    concept_category: str = Field(
        default="",
        description="from examples or create a new category (concept nodes only, otherwise empty)",
    )
    intervention_lifecycle: str = Field(
        default="", description="1-6 (only for intervention nodes)"
    )
    intervention_maturity: str = Field(
        default="", description="1-4 (only for intervention nodes)"
    )

    model_config = ConfigDict(extra="forbid")


class Edge(BaseModel):
    type: str = Field(
        min_length=1, max_length=64, description="relationship label verb"
    )
    source_node: str = Field(min_length=1, description="source node name")
    target_node: str = Field(min_length=1, description="target node name")
    description: str = Field(
        min_length=1, description="concise description of logical connection"
    )
    edge_confidence: int = Field(ge=1, le=5, description="1-5")

    model_config = ConfigDict(extra="forbid")


class LogicalChain(BaseModel):
    title: str = Field(
        ..., description="concise natural-language description of logical chain"
    )
    edges: List[Edge] = Field(..., description="edges in the logical chain")
    rationale: str = Field(..., description="rationale for the logical chain")

    model_config = ConfigDict(extra="forbid")


class Summary(BaseModel):
    summary: str = Field(..., description="Robust summary of the findings of the paper")
    limitations: str = Field(
        ...,
        description="Summary of limitations, uncertainties or identified gaps in the paper",
    )
    inference_strategy: str = Field(
        ..., description="Inference Strategy used and rationale"
    )

    model_config = ConfigDict(extra="forbid")


class ExtractionSchema(BaseModel):
    nodes: List[Node] = Field(..., description="Unique Nodes extracted from the paper")
    logical_chains: List[LogicalChain] = Field(
        ..., description="Logical chains extracted from the paper"
    )
    summary: Summary = Field(..., description="Summary of the paper")

    model_config = ConfigDict(extra="forbid")
