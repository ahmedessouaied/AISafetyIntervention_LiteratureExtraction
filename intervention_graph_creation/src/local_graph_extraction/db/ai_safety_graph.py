from pathlib import Path
import json
import shutil
from falkordb import FalkorDB
from tqdm import tqdm
from typing import List, Dict, Any

from config import load_settings
from intervention_graph_creation.src.local_graph_extraction.core import Node, Edge, PaperSchema
from intervention_graph_creation.src.local_graph_extraction.db.helpers import label_for, lit


SETTINGS = load_settings()


class AISafetyGraph:
    def __init__(self) -> None:
        self.db = FalkorDB(host=SETTINGS.falkordb.host, port=SETTINGS.falkordb.port)

    # ---------- nodes ----------

    def upsert_node(self, node: Node, paper_id: str) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        label = label_for(node.type)
        # Uniqueness by (name, type) → prevents duplicates for same typed name
        g.query(
            f"MERGE (n:{label} {{name: {lit(node.name)}, type: {lit(node.type)}}}) "
            f"SET n.description = {lit(node.description)}, "
            f"n.aliases = {lit(node.aliases)}, "
            f"n.concept_category = {lit(node.concept_category)}, "
            f"n.intervention_lifecycle = {lit(node.intervention_lifecycle)}, "
            f"n.intervention_maturity = {lit(node.intervention_maturity)}, "
            f"n.paper_id = {lit(paper_id)} "
            f"RETURN n"
        )

    # ---------- edges ----------
    # Multiple edges between same nodes are allowed,
    # but for the same etype we update the existing edge (MERGE by etype).

    def upsert_edge(self, edge: Edge, paper_id: str) -> None:
        g = self.db.select_graph(SETTINGS.falkordb.graph)
        s = lit(edge.source_node)
        t = lit(edge.target_node)
        etype = lit(edge.type)

        # Ensure endpoints exist (by name only; labels may be added elsewhere)
        g.query(f"MERGE (a {{name: {s}}}) RETURN a")
        g.query(f"MERGE (b {{name: {t}}}) RETURN b")

        # One :EDGE per (a,b,etype). If exists → update props; else → create.
        g.query(
            "MATCH (a {name: " + s + "}), (b {name: " + t + "}) "
            "MERGE (a)-[r:EDGE {etype: " + etype + "}]->(b) "
            "SET r.description = " + lit(edge.description) + ", "
            "    r.edge_confidence = " + lit(edge.edge_confidence) + ", "
            "    r.paper_id = " + lit(paper_id) + " "
            "RETURN r"
        )

    # ---------- ingest ----------

    def ingest_file(self, json_path: Path, errors: dict) -> bool:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        doc = PaperSchema(**data)

        names = [n.name for n in doc.nodes]
        dupes = sorted({x for x in names if names.count(x) > 1})
        known = set(names)
        missing = [
            (e.source_node, e.target_node)
            for ch in doc.logical_chains
            for e in ch.edges
            if e.source_node not in known or e.target_node not in known
        ]

        has_issue = False
        if dupes or missing:
            has_issue = True
            errs = []
            if dupes:
                errs.append(f"Duplicate node names: {dupes}")
            if missing:
                errs.append(f"Edges reference unknown nodes: {missing[:5]}...")
            errors[json_path.stem] = errs

        paper_id = json_path.stem
        for n in doc.nodes:
            self.upsert_node(n, paper_id)
        for ch in doc.logical_chains:
            for e in ch.edges:
                if e.source_node in known and e.target_node in known:
                    self.upsert_edge(e, paper_id)

        return has_issue

    def ingest_dir(self, input_dir: Path = SETTINGS.paths.output_dir) -> None:
        errors = {}
        base = Path(input_dir)
        issues_dir = base / "issues"
        issues_dir.mkdir(exist_ok=True)
        subdirs = [d for d in base.iterdir() if d.is_dir()]

        for d in tqdm(sorted(subdirs)):
            json_path = d / f"{d.name}.json"
            if not json_path.exists():
                print(f"⚠️ Skipping {d.name}: {json_path} not found")
                continue

            has_issue = self.ingest_file(json_path, errors)
            if has_issue:
                target_dir = issues_dir / d.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(d), str(issues_dir))
                err_file = target_dir / "errors.txt"
                err_file.write_text("\n".join(errors[d.name]), encoding="utf-8")

        if errors:
            print("\n=== Files with issues ===")
            for k, v in errors.items():
                print(f"- {k}.json: {', '.join(v)}")

    # ---------- utils ----------

    def get_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        g = self.db.select_graph(SETTINGS.falkordb.graph)

        node_res = g.ro_query("MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n AS node")
        nodes = []
        for row in node_res.result_set:
            node_id = row[0]
            labels = row[1] or []
            node = row[2]
            props = node.properties or {}
            parts = []
            for k, v in props.items():
                if k == "id":
                    continue
                if isinstance(v, str):
                    v_str = v
                elif isinstance(v, (list, tuple)):
                    v_str = ", ".join(str(x) for x in v)
                else:
                    v_str = str(v)
                if v_str:
                    parts.append(f"{k}={v_str}")
            text = "; ".join(parts) if parts else ""
            nodes.append({"id": node_id, "labels": labels, "text": text})

        edge_res = g.ro_query(
            "MATCH (n)-[r]->(m) RETURN ID(r) AS id, TYPE(r) AS type, ID(n) AS source, ID(m) AS target, r AS rel"
        )
        edges = []
        for row in edge_res.result_set:
            edge_id = row[0]
            edge_type = row[1]
            source = row[2]
            target = row[3]
            rel = row[4]
            props = rel.properties or {}
            edges.append(
                {
                    "id": edge_id,
                    "type": edge_type,
                    "source": source,
                    "target": target,
                    "properties": props,
                }
            )

        return {"nodes": nodes, "edges": edges}

    def save_graph_to_json(self, filepath: str) -> None:
        data = self.get_graph()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def merge_nodes(self, keep_name: str, remove_name: str):
        """
        Merge two nodes identified by name.
        Moves all relationships from remove_name -> keep_name, then deletes remove_name.
        """
        graph = self.db.select_graph(SETTINGS.falkordb.graph)

        q = f"""
        MATCH (n {{name: {lit(remove_name)}}})
        OPTIONAL MATCH (n)-[r]->() RETURN DISTINCT type(r) AS t
        UNION
        MATCH (n {{name: {lit(remove_name)}}})
        OPTIONAL MATCH ()-[r]->(n) RETURN DISTINCT type(r) AS t
        """
        result = graph.query(q)
        rel_types = [r[0] for r in result.result_set if r[0] is not None]

        if not rel_types:
            return graph.query(f"MATCH (a {{name: {lit(remove_name)}}}) DELETE a")

        parts = []
        for rtype in rel_types:
            parts.append(f"""
            OPTIONAL MATCH (a {{name: {lit(remove_name)}}})-[r:{rtype}]->(m)
            MATCH (b {{name: {lit(keep_name)}}})
            FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                MERGE (b)-[r2:{rtype}]->(m)
                SET r2 += r
            )
            WITH a, b
            OPTIONAL MATCH (m2)-[s:{rtype}]->(a {{name: {lit(remove_name)}}})
            FOREACH (_ IN CASE WHEN m2 IS NULL THEN [] ELSE [1] END |
                MERGE (m2)-[s2:{rtype}]->(b)
                SET s2 += s
            )
            WITH a, b
            """)

        merge_query = f"""
        MATCH (a {{name: {lit(remove_name)}}}), (b {{name: {lit(keep_name)}}})
        {"".join(parts)}
        DELETE a
        """
        return graph.query(merge_query)


def main():
    graph = AISafetyGraph()
    graph.ingest_dir(SETTINGS.paths.output_dir)
    graph.save_graph_to_json("graph.json")


if __name__ == "__main__":
    main()
