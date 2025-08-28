import json
import gzip
from pathlib import Path
from typing import Iterator
import numpy as np

def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def label_for(node_type: str) -> str:
    return "Concept" if node_type == "concept" else "Intervention"

def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over JSONL file, handling both regular and gzipped files."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def vector_to_string(v):
    """Convert numpy array to FalkorDB vector string format."""
    if v is None:
        return "[]"
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(str(float(x)) for x in v.flatten()) + "]"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(float(x)) for x in v) + "]"
    return "[]"