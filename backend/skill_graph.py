"""
skill_graph.py
--------------
Builds an in-memory skill graph from indexed contributor data and
renders it as an interactive HTML network using pyvis.

Graph structure
---------------
  Nodes:
    - Developer nodes  (type="developer")
    - Skill/language nodes (type="skill")

  Edges:
    - Developer ──uses──▶ Skill   (weight = contribution count)

The graph is built from whatever is currently in the ChromaDB
contributor-profile index (vector_store.py).  It can also accept
a plain list of contributor dicts (e.g. from seed data) directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyvis.network import Network

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_skill_graph(contributors: list[dict[str, Any]]) -> tuple[list, list]:
    """
    Convert a list of contributor dicts into graph nodes and edges.

    Returns
    -------
    nodes : list of dicts  {id, label, type, size, title}
    edges : list of dicts  {source, target, weight, title}
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()

    for c in contributors:
        dev_id = f"dev::{c['login']}"
        contributions = int(c.get("contributions", 1))

        # Developer node — size scales with contribution count
        if dev_id not in nodes:
            nodes[dev_id] = {
                "id": dev_id,
                "label": c["login"],
                "type": "developer",
                "size": max(10, min(50, contributions // 10 + 10)),
                "title": (
                    f"<b>{c['login']}</b><br>"
                    f"Contributions: {contributions}<br>"
                    f"GitHub: {c.get('html_url', '—')}"
                ),
            }

        for lang in c.get("languages", []):
            skill_id = f"skill::{lang}"

            # Skill node
            if skill_id not in nodes:
                nodes[skill_id] = {
                    "id": skill_id,
                    "label": lang,
                    "type": "skill",
                    "size": 15,
                    "title": f"<b>{lang}</b>",
                }

            # Edge (deduplicated)
            edge_key = (dev_id, skill_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": dev_id,
                        "target": skill_id,
                        "weight": contributions,
                        "title": f"{c['login']} → {lang} ({contributions} contributions)",
                    }
                )

    # Update skill-node size based on how many developers use it
    skill_degree: dict[str, int] = {}
    for e in edges:
        skill_degree[e["target"]] = skill_degree.get(e["target"], 0) + 1
    for nid, degree in skill_degree.items():
        if nid in nodes:
            nodes[nid]["size"] = max(12, min(40, degree * 8))

    return list(nodes.values()), edges


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PALETTE = {
    "developer": {"color": "#3b82d4", "shape": "dot"},
    "skill": {"color": "#7c5cd8", "shape": "diamond"},
}


def render_skill_graph_html(
    contributors: list[dict[str, Any]],
    height: str = "600px",
    output_path: str | None = None,
) -> str:
    """
    Build the skill graph and return a self-contained HTML string
    suitable for embedding in Streamlit via `st.components.v1.html()`.

    Parameters
    ----------
    contributors : list of contributor dicts (login, languages, contributions, …)
    height       : CSS height for the network canvas
    output_path  : if given, also write the HTML to this file path

    Returns
    -------
    HTML string of the interactive pyvis network.
    """
    nodes, edges = build_skill_graph(contributors)

    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#1f2328")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for node in nodes:
        palette = _PALETTE.get(node["type"], {})
        net.add_node(
            node["id"],
            label=node["label"],
            size=node["size"],
            title=node["title"],
            color=palette.get("color", "#aaaaaa"),
            shape=palette.get("shape", "dot"),
        )

    for edge in edges:
        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge["title"],
            width=max(1, edge["weight"] // 100),
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        return Path(output_path).read_text(encoding="utf-8")

    # Write to a temp file then read back (pyvis requires a file path)
    tmp = Path("data/.graph_tmp.html")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(tmp))
    html = tmp.read_text(encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return html


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def get_graph_stats(contributors: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a lightweight summary dict for display in the UI."""
    nodes, edges = build_skill_graph(contributors)
    developers = [n for n in nodes if n["type"] == "developer"]
    skills = [n for n in nodes if n["type"] == "skill"]

    # Top skills by number of developers who use them
    skill_degree: dict[str, int] = {}
    for e in edges:
        skill_degree[e["target"]] = skill_degree.get(e["target"], 0) + 1
    top_skills = sorted(skill_degree.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "num_developers": len(developers),
        "num_skills": len(skills),
        "num_edges": len(edges),
        "top_skills": [
            {"skill": s.replace("skill::", ""), "count": c} for s, c in top_skills
        ],
    }
